# ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sqlite3
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    filename='ml_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CryptoML:
    def __init__(self):
        self.model = None
        self.conn = sqlite3.connect('historical_data.db')
        self._create_tables()
        
    def _create_tables(self):
        """Initialize database tables"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_features (
                    timestamp INTEGER PRIMARY KEY,
                    symbol TEXT,
                    timeframe TEXT,
                    price REAL,
                    rsi REAL,
                    ema20 REAL,
                    ema50 REAL,
                    macd REAL,
                    adx REAL,
                    bb_width REAL,
                    liq_impact REAL,
                    volume REAL,
                    vwap REAL,
                    fib_236 REAL,
                    fib_618 REAL,
                    swing_high REAL,
                    swing_low REAL,
                    next_5m_return REAL,
                    next_1h_return REAL
                )
            ''')
            self.conn.commit()
        except Exception as e:
            logging.error(f"Database error: {str(e)}")

    def save_features(self, features):
        """Save features to database"""
        try:
            df = pd.DataFrame([features])
            df.to_sql('historical_features', self.conn, if_exists='append', index=False)
            self.conn.commit()
        except Exception as e:
            logging.error(f"Feature save error: {str(e)}")

    def preprocess_data(self, timeframe='5m', lookback_days=30):
        """Prepare training data"""
        try:
            query = f'''
                SELECT * FROM historical_features 
                WHERE timeframe = '{timeframe}' 
                AND timestamp >= {int((datetime.now() - timedelta(days=lookback_days)).timestamp()*1000)}
            '''
            data = pd.read_sql_query(query, self.conn)
            
            if len(data) < 1000:
                return None, None, None, None
                
            data['target'] = np.where(data['next_5m_return' if timeframe == '5m' else 'next_1h_return'] > 0, 1, 0)
            features = data[['rsi', 'ema20', 'ema50', 'macd', 'adx', 'bb_width', 
                            'liq_impact', 'volume', 'vwap', 'fib_236', 'fib_618',
                            'swing_high', 'swing_low']]
            labels = data['target']
            
            return train_test_split(features, labels, test_size=0.2, shuffle=False)
        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return None, None, None, None

    def train_model(self, timeframe='5m'):
        """Train and save ML model"""
        try:
            X_train, X_test, y_train, y_test = self.preprocess_data(timeframe)
            if X_train is None:
                return 0.0
                
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Add metadata
            self.model.metadata = {
                'accuracy': float(accuracy),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model
            joblib.dump(self.model, f'model_{timeframe}.joblib')
            return accuracy
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            return 0.0

    def predict_confidence(self, current_features, timeframe='5m'):
        """Make prediction and return confidence metrics"""
        try:
            model_path = f'model_{timeframe}.joblib'
            self.model = joblib.load(model_path)
            
            # Add context features
            context_features = {
                'volatility': current_features['bb_width'] * 100,
                'volume_ratio': (current_features['volume'] / 
                               self.conn.execute(f'''
                                   SELECT AVG(volume) 
                                   FROM historical_features 
                                   WHERE timeframe='{timeframe}'
                               ''').fetchone()[0]),
                'price_vs_vwap': current_features['price'] / current_features['vwap']
            }
            
            full_features = {**current_features, **context_features}
            df = pd.DataFrame([full_features])
            
            proba = self.model.predict_proba(df)[0]
            confidence = float(np.max(proba)*100)
            uncertainty = float((1 - (np.max(proba) - np.min(proba)))*100)

            return {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'suggested_width': 0.5 * uncertainty/100 * current_features.get('atr', 0.0)
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {'confidence': 50.0, 'uncertainty': 100.0, 'suggested_width': 0.0}  # Changed None to 0.0

    def get_data_counts(self):
        """Get record counts per timeframe"""
        try:
            cursor = self.conn.cursor()
            return {
                '5m': cursor.execute("SELECT COUNT(*) FROM historical_features WHERE timeframe='5m'").fetchone()[0],
                '1h': cursor.execute("SELECT COUNT(*) FROM historical_features WHERE timeframe='1h'").fetchone()[0]
            }
        except Exception as e:
            logging.error(f"Data count error: {str(e)}")
            return {}

    def get_model_info(self):
        """Get model metadata"""
        info = {}
        for tf in ['5m', '1h']:
            try:
                model = joblib.load(f'model_{tf}.joblib')
                info[tf] = {
                    'accuracy': model.metadata.get('accuracy', 0),
                    'last_trained': model.metadata.get('timestamp', 'Never')
                }
            except Exception as e:
                logging.error(f"Model info error ({tf}): {str(e)}")
                continue
        return info
