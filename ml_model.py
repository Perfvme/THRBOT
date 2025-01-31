import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import os

class MLSystem:
    def __init__(self):
        self.model = None
        self.features = [
            'RSI', 'EMA_20', 'EMA_50', 'MACD', 'VWAP', 'ADX',
            'funding_rate', 'open_interest', 'LIQUIDATION_IMPACT',
            'volatility_4h', 'EMA_diff', 'RSI_vol_corr'
        ]
        
    def feature_engineering(self, df):
        """Create features without Dask"""
        df = df.copy()
        for lag in [1, 3]:
            df[f'RSI_lag{lag}'] = df['RSI'].shift(lag)
        df['log_ret'] = np.log(df['close']).diff()
        df['volatility_4h'] = df['log_ret'].rolling(12).std()
        df['EMA_diff'] = df['EMA_20'] - df['EMA_50']
        df['RSI_vol_corr'] = df['RSI'].rolling(6).corr(df['volatility_4h'])
        return df.dropna()

    def train(self):
        """Simplified training using pandas"""
        try:
            if not os.path.exists('data/processed.parquet'):
                return False
                
            df = pd.read_parquet('data/processed.parquet')
            df = self.feature_engineering(df)
            
            if len(df) < 5000:
                return False

            X = df[self.features]
            y = df['target']

            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.1,
                num_leaves=31,
                max_depth=4,
                n_jobs=2,  # Use 2 CPU threads
                random_state=42
            )

            tscv = TimeSeriesSplit(n_splits=3)
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
        """Prediction with fallback"""
        try:
            if not self.model:
                try:
                    self.model = joblib.load('ml_model.pkl')
                except:
                    return {'confidence': 50.0, 'uncertainty': 100.0}
                
            proba = self.model.predict_proba(pd.DataFrame([current_data]))[0]
            return {
                'confidence': float(proba[1]) * 100,
                'uncertainty': float(np.abs(proba[1] - proba[0])) * 50
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {'confidence': 50.0, 'uncertainty': 100.0}
ml = MLSystem()
