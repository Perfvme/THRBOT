# analysis.py
import pandas as pd
from finta import TA
import numpy as np

def analyze_data(df, symbol):
    try:
        # Ensure numeric conversion first
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.ffill()

        # Calculate EMAs with validation
        ema_periods = [20, 50]
        for period in ema_periods:
            try:
                df[f'EMA_{period}'] = TA.EMA(df, period)
            except Exception as e:
                print(f"Error calculating EMA {period}: {str(e)}")
                df[f'EMA_{period}'] = np.nan  # Create empty column

        # Core Indicators
        df['RSI'] = TA.RSI(df).rolling(3).mean()
        df['EMA_20'] = TA.EMA(df, 20)
        df['EMA_50'] = TA.EMA(df, 50)
        df['MACD'] = TA.MACD(df)['MACD']
        df['ADX'] = TA.ADX(df)
        df['ATR'] = TA.ATR(df, 14)
        
        # Volatility
        bb = TA.BBANDS(df)
        df['BB_UPPER'] = bb['BB_UPPER']
        df['BB_LOWER'] = bb['BB_LOWER']
        
        # Volume Analysis
        df['OBV'] = TA.OBV(df)
        df['VWAP'] = TA.VWAP(df)
        
         latest = df.iloc[-1]
        return {
            'price': latest['close'],
            'rsi': latest.get('RSI', 50),
            'ema': latest.get('EMA_20', latest['close']),  # Fallback to close price
            'ema50': latest.get('EMA_50', latest['close']),
            'macd': latest['MACD'],
            'adx': latest['ADX'],
            'atr': latest['ATR'],
            'bb_upper': latest['BB_UPPER'],
            'bb_lower': latest['BB_LOWER'],
            'obv': latest['OBV'],
            'vwap': latest['VWAP']
        }
    except Exception as e:
        return {'error': f"Analysis failed: {str(e)}"}
