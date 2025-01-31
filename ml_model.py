import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
import threading

class MLModel:
    def __init__(self, model_path="ml_model.pkl"):
        self.model_path = model_path
        self.model = None
        try:
            self.load_model()
        except Exception as e:
            print(f"Error initializing the model: {e}")
            raise

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Loaded pre-trained model.")
        except (FileNotFoundError, pickle.UnpicklingError):
            print("No model found. Initializing a new model.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def retrain_model(self, data: pd.DataFrame):
        """Train the model on new data."""
        features = ['RSI', 'EMA_20', 'EMA_50', 'MACD', 'ADX', 'ATR', 'VWAP', 'BB_WIDTH']
        X = data[features]
        y = (data['price'].shift(-1) > data['price']).astype(int)  # 1 for price increase, 0 for decrease
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained. Accuracy: {accuracy:.2f}")

        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved.")

    def predict(self, data: pd.DataFrame):
        """Predict market direction (Bullish: 1, Bearish: 0)."""
        features = ['RSI', 'EMA_20', 'EMA_50', 'MACD', 'ADX', 'ATR', 'VWAP', 'BB_WIDTH']
        X = data[features]
        prediction = self.model.predict(X.iloc[[-1]])  # Predict based on the latest data point
        return prediction[0]  # Return predicted class (1 or 0)

    def start_periodic_update(self, data, interval=86400):
        def periodic_task():
            while True:
                self.update_model_periodically(data, interval)
                time.sleep(3600)

        update_thread = threading.Thread(target=periodic_task, daemon=True)
        update_thread.start()

    def update_model_periodically(self, data: pd.DataFrame, interval=86400):
        last_update_time = 0
        while True:
            current_time = time.time()
            if current_time - last_update_time >= interval:
                self.retrain_model(data)  # Retrain the model with new data
                last_update_time = current_time
            time.sleep(3600)  # Sleep for 1 hour before checking again
