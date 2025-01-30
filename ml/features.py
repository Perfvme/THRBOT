import pandas as pd
import numpy as np
from finta import TA

class FeatureEngineer:
    def __init__(self):
        self.window_size = ML_CONFIG['seq_length']
        
    def process(self, df):
        """Core feature pipeline"""
        feats = pd.DataFrame(index=df.index)
        
        # Price action
        feats['returns'] = df['close'].pct_change()
        feats['volatility'] = df['close'].rolling(20).std()
        feats['atr'] = TA.ATR(df, 14)
        
        # Volume dynamics
        feats['volume_ma'] = df['volume'].rolling(50).mean()
        feats['volume_z'] = (df['volume'] - feats['volume_ma']) / df['volume'].rolling(50).std()
        
        # Market structure
        feats['ema_cross'] = (TA.EMA(df, 20) / TA.EMA(df, 50)) - 1
        feats['rsi_divergence'] = TA.RSI(df).diff() - df['close'].diff()
        
        # Liquidation impact
        feats['liq_impact'] = df['liq_volume'] / df['volume'].replace(0, 1e-6)
        
        # Target: 1 if price rises 0.5% in next 3 periods
        feats['target'] = (df['close'].shift(-3) > df['close'] * 1.005).astype(int)
        
        return feats.dropna().iloc[-self.window_size:]