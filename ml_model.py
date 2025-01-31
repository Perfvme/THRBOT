import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import os

class MLSystem:
    def __init__(self):
        self.model = None
        self.client = Client(n_workers=1, threads_per_worker=2, memory_limit='4GB')
        self.features = [
            'RSI', 'EMA_20', 'EMA_50', 'MACD', 'VWAP', 'ADX',
            'funding_rate', 'open_interest', 'LIQUIDATION_IMPACT',
            'volatility_4h', 'EMA_diff', 'RSI_vol_corr'
        ]
        
    def feature_engineering(self, df):
        """Resource-efficient feature creation"""
        df = df.copy()
        # Lag features
        for lag in [1, 3]:
            df[f'RSI_lag{lag}'] = df['RSI'].shift(lag)
        # Volatility
        df['log_ret'] = np.log(df['close']).diff()
        df['volatility_4h'] = df['log_ret'].rolling(12).std()
        # Interactions
        df['EMA_diff'] = df['EMA_20'] - df['EMA_50']
        df['RSI_vol_corr'] = df['RSI'].rolling(6).corr(df['volatility_4h'])
        return df.dropna()

    def train(self):
        """Memory-constrained training pipeline"""
        try:
            ddf = dd.read_parquet('data/processed.parquet')
            ddf = ddf.map_partitions(self.feature_engineering)
            
            if len(ddf) < 5000:  # Minimum data check
                return False

            # Time-series validation
            tscv = TimeSeriesSplit(n_splits=3)
            X = ddf[self.features].compute()
            y = ddf['target'].compute()

            # Model config optimized for 8GB RAM
            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.1,
                num_leaves=31,
                max_depth=4,
                n_jobs=2,
                random_state=42
            )

            scores = []
            for train_idx, test_idx in tscv.split(X):
                self.model.fit(X.iloc[train_idx], y.iloc[train_idx],
                              eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
                              early_stopping_rounds=15,
                              verbose=-1)
                scores.append(roc_auc_score(y.iloc[test_idx], 
                                          self.model.predict_proba(X.iloc[test_idx])[:,1]))

            joblib.dump(self.model, 'ml_model.pkl')
            return np.mean(scores)
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False

    def predict(self, current_data):
        """RAM-friendly prediction"""
        try:
            if not self.model:
                self.model = joblib.load('ml_model.pkl')
                
            proba = self.model.predict_proba(pd.DataFrame([current_data]))[0]
            return {
                'confidence': float(proba[1]) * 100,
                'uncertainty': float(np.abs(proba[1] - proba[0])) * 50
            }
        except:
            return {'confidence': 50.0, 'uncertainty': 100.0}

ml = MLSystem()