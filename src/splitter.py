import pandas as pd
from typing import Tuple
from datetime import timedelta


class TimeSeriesSplitter:
    #Time-based data splitting
    
    def __init__(self, trainRatio: float = 0.7, validRatio: float = 0.15):
        self.trainRatio = trainRatio
        self.validRatio = validRatio
    
    def split(self, df: pd.DataFrame, dateCol: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        #Split by date
        dates = df[dateCol].sort_values().unique()
        nDates = len(dates)
        
        trainEnd = dates[int(nDates * self.trainRatio)]
        validEnd = dates[int(nDates * (self.trainRatio + self.validRatio))]
        
        train = df[df[dateCol] <= trainEnd].copy()
        valid = df[(df[dateCol] > trainEnd) & (df[dateCol] <= validEnd)].copy()
        test = df[df[dateCol] > validEnd].copy()
        
        print("Data split:")
        print(f"  Train: {train[dateCol].min().date()} to {train[dateCol].max().date()} ({len(train):,} rows)")
        print(f"  Valid: {valid[dateCol].min().date()} to {valid[dateCol].max().date()} ({len(valid):,} rows)")
        print(f"  Test:  {test[dateCol].min().date()} to {test[dateCol].max().date()} ({len(test):,} rows)")
        
        return train, valid, test
    
    def dropWarmupPeriod(self, df: pd.DataFrame, nDays: int = 28, dateCol: str = 'date') -> pd.DataFrame:
        #Drop initial rows with NaN from lags
        minDate = df[dateCol].min()
        cutoff = minDate + timedelta(days=nDays)
        dfClean = df[df[dateCol] >= cutoff].copy()
        print(f"  Dropped {len(df) - len(dfClean):,} warmup rows")
        return dfClean
    
    def prepareXY(self, df: pd.DataFrame, featureCols: list, targetCol: str = 'log_sales') -> Tuple[pd.DataFrame, pd.Series]:
        #Prepare X and y
        available = [c for c in featureCols if c in df.columns]
        X = df[available].fillna(0)
        y = df[targetCol]
        return X, y