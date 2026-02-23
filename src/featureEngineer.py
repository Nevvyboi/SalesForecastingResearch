import pandas as pd
import numpy as np
from typing import List, Dict
import pickle


class FeatureEngineer:
    #Creates features for forecasting
    
    TEMPORAL_FEATURES = [
        'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'is_month_start', 'is_month_end', 'dow_sin', 'dow_cos'
    ]
    LAG_FEATURES = ['lag_7', 'lag_14', 'lag_21', 'lag_28']
    ROLLING_FEATURES = ['rolling_mean_7', 'rolling_mean_14', 'rolling_std_7']
    EXTERNAL_FEATURES = ['is_holiday', 'onpromotion']
    ENCODED_FEATURES = ['store_id_enc', 'family_enc']
    STATIC_FEATURES = ['perishable']
    
    def __init__(self):
        self.targetEncodings: Dict[str, Dict] = {}
        self.globalMean: float = None
    
    def addTemporalFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        #Calendar features
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek.astype('int8')
        df['day_of_month'] = df['date'].dt.day.astype('int8')
        df['month'] = df['date'].dt.month.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_month_start'] = (df['day_of_month'] <= 5).astype('int8')
        df['is_month_end'] = (df['day_of_month'] >= 26).astype('int8')
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df
    
    def addLagFeatures(self, df: pd.DataFrame, targetCol: str = 'log_sales') -> pd.DataFrame:
        #Lag features
        df = df.copy()
        for lag in [7, 14, 21, 28]:
            df[f'lag_{lag}'] = df.groupby(['store_id', 'product_id'])[targetCol].shift(lag)
        return df
    
    def addRollingFeatures(self, df: pd.DataFrame, targetCol: str = 'log_sales') -> pd.DataFrame:
        #Rolling statistics
        df = df.copy()
        for window in [7, 14]:
            shifted = df.groupby(['store_id', 'product_id'])[targetCol].shift(1)
            df[f'rolling_mean_{window}'] = shifted.groupby([df['store_id'], df['product_id']]).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        shifted = df.groupby(['store_id', 'product_id'])[targetCol].shift(1)
        df['rolling_std_7'] = shifted.groupby([df['store_id'], df['product_id']]).transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )
        return df
    
    def fitTargetEncoding(self, df: pd.DataFrame, targetCol: str, catCols: List[str], smoothing: float = 10.0) -> None:
        #Fit target encodings
        self.globalMean = df[targetCol].mean()
        for col in catCols:
            if col not in df.columns:
                print(f"  Warning: {col} not found, skipping encoding")
                continue
            stats = df.groupby(col)[targetCol].agg(['mean', 'count'])
            smoothed = (stats['count'] * stats['mean'] + smoothing * self.globalMean) / (stats['count'] + smoothing)
            self.targetEncodings[col] = smoothed.to_dict()
    
    def applyTargetEncoding(self, df: pd.DataFrame) -> pd.DataFrame:
        #Apply target encodings
        df = df.copy()
        for col, mapping in self.targetEncodings.items():
            if col in df.columns:
                df[f'{col}_enc'] = df[col].map(mapping).fillna(self.globalMean)
            else:
                df[f'{col}_enc'] = self.globalMean
        return df
    
    def buildFeatures(self, df: pd.DataFrame, targetCol: str = 'log_sales', isTraining: bool = True) -> pd.DataFrame:
        #Full feature pipeline
        print("Building features...")
        df = self.addTemporalFeatures(df)
        df = self.addLagFeatures(df, targetCol)
        df = self.addRollingFeatures(df, targetCol)
        
        #Target encoding - only encode columns that exist
        encodeCols = [c for c in ['store_id', 'family'] if c in df.columns]
        if isTraining and encodeCols:
            self.fitTargetEncoding(df, targetCol, encodeCols)
        df = self.applyTargetEncoding(df)
        
        #Ensure all expected columns exist
        for col in self.ENCODED_FEATURES:
            if col not in df.columns:
                df[col] = self.globalMean if self.globalMean else 0
        
        for col in self.STATIC_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        print(f"  Created {len(self.getFeatureColumns())} features")
        return df
    
    def getFeatureColumns(self) -> List[str]:
        return self.TEMPORAL_FEATURES + self.LAG_FEATURES + self.ROLLING_FEATURES + self.EXTERNAL_FEATURES + self.ENCODED_FEATURES + self.STATIC_FEATURES
    
    def getFeatureGroups(self) -> Dict[str, List[str]]:
        return {
            'temporal': self.TEMPORAL_FEATURES,
            'lag': self.LAG_FEATURES,
            'rolling': self.ROLLING_FEATURES,
            'external': self.EXTERNAL_FEATURES,
            'encoded': self.ENCODED_FEATURES
        }
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'targetEncodings': self.targetEncodings, 'globalMean': self.globalMean}, f)
    
    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.targetEncodings = data['targetEncodings']
            self.globalMean = data['globalMean']