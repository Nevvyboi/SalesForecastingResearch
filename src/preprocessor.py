import pandas as pd
import numpy as np
from typing import Dict


class Preprocessor:
    #Cleans and merges data
    
    def cleanTarget(self, df: pd.DataFrame) -> pd.DataFrame:
        #Clip negatives and log transform
        df = df.copy()
        df['sales'] = df['sales'].clip(lower=0)
        df['log_sales'] = np.log1p(df['sales'])
        return df
    
    def mergeAll(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        #Merge all data sources
        print("Preprocessing...")
        
        df = self.cleanTarget(data['sales'])
        
        #Merge store info
        df = df.merge(data['stores'], on='store_id', how='left')
        
        #Merge product info
        df = df.merge(data['products'], on='product_id', how='left')
        
        #Merge holidays
        holidayDates = set(data['holidays']['date'])
        df['is_holiday'] = df['date'].isin(holidayDates).astype('int8')
        
        #Sort for feature engineering
        df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
        
        print(f"  Final shape: {df.shape}")
        return df