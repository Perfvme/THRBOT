# analysis.py
import pandas as pd
from finta import TA
import numpy as np

def analyze_data(df, symbol):
    try:
        df = df.ffill().apply(pd.to_numeric, errors='coerce')
        
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
            'rsi': latest['RSI'],
            'ema20': latest['EMA_20'],
            'ema50': latest['EMA_50'],
            'macd': latest['MACD'],
            'adx': latest['ADX'],
            'atr': latest['ATR'],
            'bb_upper': latest['BB_UPPER'],
            'bb_lower': latest['BB_LOWER'],
            'obv': latest['OBV'],
            'vwap': latest['VWAP']
        }
    except Exception as e:
        return {'error': str(e)}