import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import time


class SalesDataset(Dataset):
    #PyTorch Dataset for sequences
    
    def __init__(self, df: pd.DataFrame, seqLength: int = 28, targetCol: str = 'log_sales', featureCols: List[str] = None):
        self.seqLength = seqLength
        
        if featureCols is None:
            featureCols = [targetCol, 'day_of_week', 'is_holiday', 'onpromotion']
        
        self.featureCols = [c for c in featureCols if c in df.columns]
        self.sequences = []
        self.targets = []
        
        for (store, product), group in df.groupby(['store_id', 'product_id']):
            group = group.sort_values('date')
            values = group[self.featureCols].values
            target = group[targetCol].values
            
            for i in range(len(group) - seqLength):
                self.sequences.append(values[i:i + seqLength])
                self.targets.append(target[i + seqLength])
        
        self.sequences = np.nan_to_num(np.array(self.sequences, dtype=np.float32), 0)
        self.targets = np.nan_to_num(np.array(self.targets, dtype=np.float32), 0)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


class LSTMForecaster(nn.Module):
    #LSTM model
    
    def __init__(self, inputSize: int = 4, hiddenSize: int = 64, numLayers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True, 
                           dropout=dropout if numLayers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hiddenSize, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def trainLstm(model, trainLoader, validLoader, numEpochs: int = 30, lr: float = 0.001, 
              patience: int = 5, device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'valid_loss': []}
    bestLoss = float('inf')
    patienceCount = 0
    bestState = None
    startTime = time.time()
    
    for epoch in range(numEpochs):
        #Train
        model.train()
        trainLosses = []
        for xBatch, yBatch in trainLoader:
            xBatch, yBatch = xBatch.to(device), yBatch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xBatch), yBatch)
            loss.backward()
            optimizer.step()
            trainLosses.append(loss.item())
        
        #Valid
        model.eval()
        validLosses = []
        with torch.no_grad():
            for xBatch, yBatch in validLoader:
                xBatch, yBatch = xBatch.to(device), yBatch.to(device)
                validLosses.append(criterion(model(xBatch), yBatch).item())
        
        trainLoss = np.mean(trainLosses)
        validLoss = np.mean(validLosses)
        history['train_loss'].append(trainLoss)
        history['valid_loss'].append(validLoss)
        
        if validLoss < bestLoss:
            bestLoss = validLoss
            patienceCount = 0
            bestState = model.state_dict().copy()
        else:
            patienceCount += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{numEpochs} | Train: {trainLoss:.4f} | Valid: {validLoss:.4f}")
        
        if patienceCount >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(bestState)
    history['total_time'] = time.time() - startTime
    history['best_epoch'] = len(history['train_loss']) - patienceCount
    
    return model, history