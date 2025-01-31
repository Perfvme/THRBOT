import pandas as pd
import numpy as np
import os

class DataCollector:
    def __init__(self):
        os.makedirs('data/raw', exist_ok=True)
        
    def store_data(self, df, symbol, timeframe):
        """Store data with memory checks"""
        try:
            # Create target (1 if next close > current close)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df = df.dropna()
            
            # Add market regime features
            df['volatility_4h'] = np.log(df['close']).diff().rolling(12).std()
            df['EMA_diff'] = df['EMA_20'] - df['EMA_50']
            
            # Store in compressed format
            df.to_parquet(
                f'data/raw/{symbol}_{timeframe}.parquet',
                engine='fastparquet',
                compression='snappy'
            )
            
            # Merge into main dataset
            self._merge_data(symbol, timeframe)
            return True
        except Exception as e:
            print(f"Storage error: {str(e)}")
            return False
            
    def _merge_data(self, symbol, timeframe):
        """Merge new data with existing"""
        try:
            main_df = pd.read_parquet('data/processed.parquet') if os.path.exists('data/processed.parquet') else None
            new_df = pd.read_parquet(f'data/raw/{symbol}_{timeframe}.parquet')
            
            if main_df is not None:
                combined = pd.concat([main_df, new_df]).drop_duplicates(subset=['timestamp'])
            else:
                combined = new_df
                
            # Keep only 6 months data to save space
            combined = combined[combined['timestamp'] > (pd.Timestamp.now() - pd.DateOffset(months=6))]
            combined.to_parquet('data/processed.parquet', engine='fastparquet', compression='snappy')
        except Exception as e:
            print(f"Merge error: {str(e)}")