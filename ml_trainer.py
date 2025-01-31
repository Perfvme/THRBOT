# ml_trainer.py
import sqlite3
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import data_fetcher

class MLTrainer:
    def __init__(self, db_path='ml_data.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_table()
        self.model = LogisticRegression(max_iter=1000, random_state=42)  # Lightweight model
        self._load_model()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                timeframe TEXT,
                rsi REAL,
                ema20 REAL,
                ema50 REAL,
                macd REAL,
                adx REAL,
                bb_width REAL,
                price REAL,
                trend_bullish INTEGER,
                trend_bearish INTEGER,
                outcome INTEGER
            )
        ''')
        self.conn.commit()

    def save_analysis(self, data):
        df = pd.DataFrame([data])
        df.to_sql('analysis_data', self.conn, if_exists='append', index=False)
        self.conn.commit()

    def _get_pending_analyses(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, symbol, timeframe, timestamp, price, trend_bullish, trend_bearish 
            FROM analysis_data 
            WHERE outcome IS NULL
        ''')
        return cursor.fetchall()

    def _timeframe_to_delta(self, tf):
        deltas = {
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return deltas.get(tf, timedelta(minutes=5))

    def update_outcomes(self):
        pending = self._get_pending_analyses()
        for analysis in pending:
            id, symbol, tf, ts, price, t_bull, t_bear = analysis
            analysis_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            if datetime.now() >= analysis_time + self._timeframe_to_delta(tf):
                try:
                    df, error = data_fetcher.get_crypto_data(symbol, tf)
                    if not error and not df.empty:
                        new_price = df['close'].iloc[-1]
                        direction = 1 if new_price > price else -1  # -1 for bearish
                        expected = 1 if t_bull > t_bear else -1 if t_bear > t_bull else 0
                        outcome = 1 if direction == expected else 0
                        if outcome is not None:
                            cursor = self.conn.cursor()
                            cursor.execute('UPDATE analysis_data SET outcome=? WHERE id=?', (outcome, id))
                            self.conn.commit()
                except Exception as e:
                    print(f"Error updating outcome: {str(e)}")

    def _load_model(self):
        try:
            self.model = joblib.load('ml_model.pkl')
        except (FileNotFoundError, EOFError):
            self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT rsi, ema20, ema50, macd, adx, bb_width, price, outcome FROM analysis_data WHERE outcome IS NOT NULL')
            data = cursor.fetchall()
            if len(data) < 100: 
                print("Not enough data for training. Waiting for more samples.")
                return

            df = pd.DataFrame(data, columns=['rsi','ema20','ema50','macd','adx','bb_width','price','outcome'])
            X = df.drop('outcome', axis=1)
            y = df['outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, 'ml_model.pkl')
            
            print(f"Model trained. Accuracy: {accuracy_score(y_test, self.model.predict(X_test)):.2f}")
            
        except Exception as e:
            print(f"Training failed: {str(e)}")

    def predict(self, features):
        try:
            if not hasattr(self.model, 'coef_'):
                return 50.0  # Neutral confidence if no model
            return self.model.predict_proba([features])[0][1] * 100
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return 50.0

ml_engine = MLTrainer()
