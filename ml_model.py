# ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    filename='ml_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CryptoML:
    def __init__(self):
        self.models: Dict[str, any] = {}
        self.conn = sqlite3.connect('historical_data.db', check_same_thread=False)
        self._create_tables()
        self.parquet_path = 'historical_data.parquet.gzip'
        
    def _create_tables(self) -> None:
        """Initialize database tables with schema optimization"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_features (
                    timestamp INTEGER PRIMARY KEY,
                    symbol TEXT,
                    timeframe TEXT,
                    rsi REAL,
                    ema20 REAL,
                    ema50 REAL,
                    macd REAL,
                    adx REAL,
                    bb_width REAL,
                    liq_impact REAL,
                    next_5m_return REAL,
                    next_1h_return REAL
                ) WITHOUT ROWID
            ''')
            self.conn.execute('PRAGMA journal_mode = WAL')
            self.conn.commit()
        except Exception as e:
            logging.error(f"Database error: {str(e)}")

    def save_features(self, features: Dict) -> None:
        """Save features to database with batch optimization"""
        try:
            # Use vectorized operations for efficiency
            df = pd.DataFrame([features])
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            
            # Use replace instead of append for conflict handling
            df.to_sql('historical_features', self.conn, if_exists='append', 
                     index=False, method='multi')
        except Exception as e:
            logging.error(f"Feature save error: {str(e)}")

    def _load_data(self, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
        """Optimized data loading with caching"""
        try:
            if os.path.exists(self.parquet_path):
                data = pd.read_parquet(self.parquet_path)
                time_filter = datetime.now() - timedelta(days=lookback_days)
                return data[
                    (data['timeframe'] == timeframe) &
                    (data['timestamp'] >= time_filter.timestamp() * 1000)
                ].copy()
            
            # Fallback to SQL
            query = f'''
                SELECT * FROM historical_features 
                WHERE timeframe = '{timeframe}' 
                AND timestamp >= {int((datetime.now() - timedelta(days=lookback_days)).timestamp()*1000)}
            '''
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            return pd.DataFrame()

    def preprocess_data(self, timeframe: str, lookback_days: int = 60) -> Tuple:
        """Enhanced feature engineering with memory optimization"""
        try:
            data = self._load_data(timeframe, lookback_days)
            
            if len(data) < 1000:
                return (None,)*4

            # Feature transformations
            numeric_cols = ['rsi', 'ema20', 'ema50', 'macd', 'adx', 'bb_width', 'liq_impact']
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Generate technical features
            data['ema_diff'] = data['ema20'] - data['ema50']
            data['rsi_ema'] = data['rsi'].rolling(5, min_periods=1).mean()
            data['macd_signal_diff'] = data['macd'] - data['macd'].shift(1)
            data['bb_width_change'] = data['bb_width'].pct_change().fillna(0)
            data['liq_impact_sma'] = data['liq_impact'].rolling(8, min_periods=1).mean()
            
            # Lagged features with memory optimization
            for lag in [1, 3, 5]:
                data[f'rsi_lag{lag}'] = data['rsi'].shift(lag).astype(np.float32)
                data[f'volume_change_{lag}'] = data['volume'].pct_change(lag).astype(np.float32)
            
            # Target encoding with ternary classification
            return_col = 'next_5m_return' if timeframe == '5m' else 'next_1h_return'
            data['target'] = np.select(
                [data[return_col] > 0.005, data[return_col] < -0.005],
                [2, 0], default=1
            ).astype(np.int8)
            
            # Clean and reduce memory usage
            data = data.dropna().reset_index(drop=True)
            data = data.drop(['timestamp', 'symbol', 'timeframe', 
                            'next_5m_return', 'next_1h_return'], axis=1)
            
            # Optimize memory
            for col in data.columns:
                if data[col].dtype == np.float64:
                    data[col] = data[col].astype(np.float32)
                    
            return train_test_split(
                data.drop('target', axis=1), 
                data['target'],
                test_size=0.2,
                shuffle=False,
                stratify=data['target']
            )
        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return (None,)*4

    def get_data_counts(self) -> Dict[str, int]:
        """Get optimized record counts using WAL mode"""
        try:
            return {
                '5m': pd.read_sql("SELECT COUNT(*) FROM historical_features WHERE timeframe='5m'", 
                                 self.conn).iloc[0,0],
                '1h': pd.read_sql("SELECT COUNT(*) FROM historical_features WHERE timeframe='1h'", 
                                 self.conn).iloc[0,0]
            }
        except Exception as e:
            logging.error(f"Data count error: {str(e)}")
            return {}

    def get_model_info(self) -> Dict:
        """Get model metadata with cache"""
        info = {}
        for tf in ['5m', '1h']:
            try:
                model = joblib.load(f'model_{tf}.joblib')
                info[tf] = {
                    'accuracy': model.metadata.get('accuracy', 0.0),
                    'last_trained': model.metadata.get('timestamp', '1970-01-01'),
                    'data_points': model.metadata.get('data_points', 0)
                }
            except Exception as e:
                continue
        return info

    def train_model(self, timeframe: str) -> float:
        """Memory-optimized training with model stacking"""
        try:
            X_train, X_test, y_train, y_test = self.preprocess_data(timeframe)
            if X_train is None or len(X_train) < 5000:
                return 0.0

            # Create efficient model ensemble
            base_models = [
                ('hgb', HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_depth=7,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                    verbose=0
                )),
                ('nb', CalibratedClassifierCV(
                    GaussianNB(),
                    method='isotonic',
                    n_jobs=1
                ))
            ]

            model = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(
                    penalty='l2',
                    C=0.1,
                    solver='lbfgs',
                    max_iter=1000,
                    n_jobs=1
                ),
                cv=3,
                n_jobs=1
            )

            # Train with batch processing
            batch_size = 1000
            for i in range(0, len(X_train), batch_size):
                end_idx = min(i+batch_size, len(X_train))
                model.fit(X_train.iloc[i:end_idx], y_train.iloc[i:end_idx])

            # Calibration and evaluation
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calibrated_model.fit(X_test, y_test)
            accuracy = accuracy_score(y_test, calibrated_model.predict(X_test))

            # Save with optimized storage
            calibrated_model.metadata = {
                'accuracy': float(accuracy),
                'timestamp': datetime.now().isoformat(),
                'data_points': len(X_train),
                'feature_importance': dict(
                    zip(X_train.columns, model.final_estimator_.coef_[0])
                )
            }
            
            joblib.dump(
                calibrated_model, 
                f'model_{timeframe}.joblib',
                compress=3,
                protocol=4
            )
            
            return accuracy
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            return 0.0

    def predict_confidence(self, current_features: Dict, timeframe: str) -> float:
        """Efficient prediction with model cache"""
        try:
            if timeframe not in self.models:
                self.models[timeframe] = joblib.load(f'model_{timeframe}.joblib')
                
            model = self.models[timeframe]
            df = pd.DataFrame([current_features])
            
            # Feature alignment
            for col in model.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0.0
                    
            df = df[model.feature_names_in_].astype(np.float32)
            proba = model.predict_proba(df)
            return float(np.max(proba[0]) * 100)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 50.0

    def optimize_storage(self) -> None:
        """Convert SQLite data to compressed Parquet format"""
        try:
            df = pd.read_sql('SELECT * FROM historical_features', self.conn)
            df.to_parquet(
                self.parquet_path,
                engine='pyarrow',
                compression='gzip',
                index=False
            )
            self.conn.execute('DELETE FROM historical_features')
            self.conn.commit()
        except Exception as e:
            logging.error(f"Storage optimization failed: {str(e)}")

    def incremental_learning(self, timeframe: str) -> bool:
        """Update model with new data efficiently"""
        try:
            model = joblib.load(f'model_{timeframe}.joblib')
            new_data = self._load_data(timeframe, lookback_days=1)
            
            if len(new_data) < 100:
                return False

            X_new = self.preprocess_data(timeframe, lookback_days=1)[0]
            if X_new is None or len(X_new) == 0:
                return False

            # Partial fit if supported
            if hasattr(model, 'partial_fit'):
                model.partial_fit(
                    X_new,
                    new_data['target'],
                    classes=[0, 1, 2]
                )
                joblib.dump(model, f'model_{timeframe}.joblib')
                return True
            return False
        except Exception as e:
            logging.error(f"Incremental learning failed: {str(e)}")
            return False

    def detect_concept_drift(self, timeframe: str, window_size: int = 5000) -> bool:
        """Lightweight drift detection using accuracy rolling window"""
        try:
            data = pd.read_parquet(self.parquet_path)
            recent_data = data[data['timeframe'] == timeframe].tail(window_size)
            
            if len(recent_data) < 1000:
                return False
                
            # Calculate accuracy decay
            baseline = self.get_model_info().get(timeframe, {}).get('accuracy', 60)
            recent_correct = (model.predict(recent_data) == recent_data['target']).mean()
            
            if recent_correct < (baseline * 0.95):  # 5% accuracy drop
                self.train_model(timeframe)
                return True
            return False
        except Exception as e:
            logging.error(f"Drift detection failed: {str(e)}")
            return False
