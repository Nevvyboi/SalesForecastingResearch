import numpy as np
import pandas as pd
from typing import Dict


class Evaluator:
    #Forecasting metrics
    
    @staticmethod
    def rmsle(yTrue, yPred) -> float:
        return np.sqrt(np.mean((yTrue - yPred) ** 2))
    
    @staticmethod
    def mae(yTrue, yPred) -> float:
        return np.mean(np.abs(yTrue - yPred))
    
    def computeMetrics(self, yTrue, yPred) -> Dict[str, float]:
        if isinstance(yPred, np.ndarray):
            yPred = pd.Series(yPred)
        if isinstance(yTrue, np.ndarray):
            yTrue = pd.Series(yTrue)
        
        mask = ~(yTrue.isna() | yPred.isna())
        yTrue, yPred = yTrue[mask], yPred[mask]
        
        return {
            'rmsle': self.rmsle(yTrue, yPred),
            'mae': self.mae(yTrue, yPred),
            'n_samples': len(yTrue)
        }