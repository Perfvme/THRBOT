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
            'funding_rate', 'open_interest', 'LIQUIDATION_IMPACT'
        ]
        
    def create_features(self, df):
        """Feature engineering without multiprocessing"""
        df = df.copy()
        # Lagged indicators
        df['RSI_1'] = df['RSI'].shift(1)
        df['RSI_3'] = df['RSI'].shift(3)
        # Volatility
        df['log_ret'] = np.log(df['close']).diff()
        df['volatility'] = df['log_ret'].rolling(12).std()
        return df.dropna()

    def train(self):
        """Simplified training process"""
        try:
            if not os.path.exists('data/processed.parquet'):
                return False
                
            df = pd.read_parquet('data/processed.parquet')
            df = self.create_features(df)
            
            if len(df) < 1000:
                return False

            X = df[self.features]
            y = df['target']

            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                n_jobs=1,  # Single-threaded to avoid multiprocessing
                random_state=42
            )

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, test_idx in tscv.split(X):
                self.model.fit(X.iloc[train_idx], y.iloc[train_idx])
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
