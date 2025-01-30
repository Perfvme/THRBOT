import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from .model import CryptoModel
from .features import FeatureEngineer

class ModelTrainer:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def fetch_training_data(self):
        dfs = []
        for sym in ML_CONFIG['train_symbols']:
            df, _ = data_fetcher.get_crypto_data(sym, '1m')
            dfs.append(self.fe.process(df))
        return pd.concat(dfs)
    
    def async_train(self):
        """Non-blocking training"""
        self.executor.submit(self._train_impl)
        
    def _train_impl(self):
        data = self.fetch_training_data()
        X = data.drop('target', axis=1)
        y = data['target']
        
        model = CryptoModel()
        model.train(X, y)