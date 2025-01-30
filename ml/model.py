# ml_model.py
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

class MLEngine:
    def __init__(self):
        self.model = make_pipeline(
            MinMaxScaler(),
            LGBMClassifier(n_estimators=150, num_leaves=15, n_jobs=1)
        )
        
    def train_initial_model(self):
        try:
            df = pd.read_csv('sample_data.csv')  # Pre-collected data
            X = df.drop('target', axis=1)
            y = df['target']
            self.model.fit(X, y)
            joblib.dump(self.model, 'ml_model.pkl')
        except:
            pass
            
    def predict(self, df):
        try:
            features = pd.DataFrame({
                'rsi': df['RSI'].iloc[-1],
                'ema_ratio': df['EMA_20'].iloc[-1] / df['EMA_50'].iloc[-1],
                'macd_hist': df['MACD'].iloc[-1],
                'atr': df['ATR'].iloc[-1],
                'volume_change': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            }, index=[0])
            
            proba = self.model.predict_proba(features)[0][1]
            return {'confidence': proba * 100}
        except:
            return {'confidence': 50.0}  # Neutral fallback