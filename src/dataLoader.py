import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


class DataLoader:
    #Loads UCI Online Retail dataset
    
    def __init__(self, dataPath: str = "data/"):
        self.dataPath = Path(dataPath)
    
    def load(self, nRows: int = None) -> pd.DataFrame:
        #Load and prepare UCI Online Retail data
        print("Loading UCI Online Retail dataset...")
        
        #Try different file names
        possibleFiles = ['online_retail.csv', 'Online Retail.csv', 'OnlineRetail.csv']
        filePath = None
        for f in possibleFiles:
            if (self.dataPath / f).exists():
                filePath = self.dataPath / f
                break
        
        if filePath is None:
            raise FileNotFoundError(
                f"Dataset not found in {self.dataPath}\n"
                "Run: python downloadData.py"
            )
        
        #Load CSV
        df = pd.read_csv(filePath, nrows=nRows, encoding='latin1')
        
        #Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        #Parse date
        if 'invoicedate' in df.columns:
            df['date'] = pd.to_datetime(df['invoicedate'], format='mixed', dayfirst=True)
        elif 'invoice_date' in df.columns:
            df['date'] = pd.to_datetime(df['invoice_date'], format='mixed', dayfirst=True)
        
        #Get quantity and price columns
        qtyCol = 'quantity' if 'quantity' in df.columns else None
        priceCol = 'unitprice' if 'unitprice' in df.columns else 'unit_price' if 'unit_price' in df.columns else None
        
        if qtyCol and priceCol:
            df['sales'] = df[qtyCol] * df[priceCol]
        
        #Clean data
        df = df[df['sales'] > 0]
        df = df[df[qtyCol] > 0]
        
        #Create store_id from country
        if 'country' in df.columns:
            df['store_id'] = df['country'].astype('category').cat.codes + 1
        else:
            df['store_id'] = 1
        
        #Create product_id from stockcode
        stockCol = 'stockcode' if 'stockcode' in df.columns else 'stock_code' if 'stock_code' in df.columns else None
        if stockCol:
            df['product_id'] = df[stockCol].astype('category').cat.codes + 1
        else:
            df['product_id'] = 1
        
        #Create family from description (first word or category)
        if 'description' in df.columns:
            df['family'] = df['description'].fillna('UNKNOWN').str.split().str[0].str.upper()
            #Limit to top 20 families + OTHER
            topFamilies = df['family'].value_counts().head(20).index.tolist()
            df['family'] = df['family'].apply(lambda x: x if x in topFamilies else 'OTHER')
        else:
            df['family'] = 'RETAIL'
        
        #Aggregate to daily sales per store-product
        print("  Aggregating to daily level...")
        daily = df.groupby(['date', 'store_id', 'product_id', 'family']).agg({
            'sales': 'sum',
            qtyCol: 'sum'
        }).reset_index()
        daily = daily.rename(columns={qtyCol: 'quantity'})
        
        #Add missing columns
        daily['perishable'] = 0
        daily['onpromotion'] = 0
        
        print(f"  Loaded: {len(daily):,} rows")
        print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
        print(f"  Stores: {daily['store_id'].nunique()}")
        print(f"  Products: {daily['product_id'].nunique()}")
        print(f"  Families: {daily['family'].nunique()}")
        
        return daily
    
    def loadAll(self, nRows: int = None) -> Dict[str, pd.DataFrame]:
        #Load all data into dict format
        df = self.load(nRows)
        
        #Create auxiliary tables
        stores = df[['store_id']].drop_duplicates()
        stores['store_type'] = 'A'
        
        products = df[['product_id', 'family', 'perishable']].drop_duplicates()
        
        #UK bank holidays
        holidays = pd.DataFrame({
            'date': pd.to_datetime(['2010-12-25', '2010-12-26', '2010-12-27', '2011-01-01', 
                                    '2011-04-22', '2011-04-25', '2011-05-02', '2011-05-30',
                                    '2011-08-29', '2011-12-25', '2011-12-26', '2011-12-27']),
            'holiday': ['Christmas', 'Boxing Day', 'Bank Holiday', 'New Year', 
                       'Good Friday', 'Easter Monday', 'May Day', 'Spring Bank',
                       'Summer Bank', 'Christmas', 'Boxing Day', 'Bank Holiday']
        })
        
        return {
            'sales': df,
            'stores': stores,
            'products': products,
            'holidays': holidays
        }