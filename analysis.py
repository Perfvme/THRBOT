# analysis.py
import pandas as pd
from finta import TA
import data_fetcher

def analyze_data(df, symbol):
    try:
        # Convert all dataframe columns to numeric first
        df = df.apply(pd.to_numeric, errors='coerce').ffill()
        
        # Enhanced indicators with explicit type conversion
        df['RSI'] = pd.to_numeric(TA.RSI(df), errors='coerce').rolling(3).mean()
        
        # Add futures market analysis
        futures_data = data_fetcher.get_futures_data(symbol)
        if futures_data:
            df['funding_rate'] = futures_data['funding_rate']
            df['open_interest'] = futures_data['open_interest']
            # Enhanced liquidation analysis
            liquidations = futures_data.get('liquidations', [])
            liquidation_prices = pd.Series([liq['price'] for liq in liquidations])
            df['LIQ_CLUSTERS'] = liquidation_prices.value_counts().reindex(df.index, fill_value=0)
            df['LIQ_SIDE_RATIO'] = pd.Series([1 if liq['side'] == 'long' else -1 for liq in liquidations]).rolling(4).mean()
            # Initialize LIQ_VOLUME with zeros first
            df['LIQ_VOLUME'] = 0
            if liquidations:
                liq_volume_series = pd.Series([liq.get('amount', 0) for liq in liquidations])
                df['LIQ_VOLUME'] = liq_volume_series.rolling(8, min_periods=1).sum().reindex(df.index, fill_value=0).ffill()
        else:
            df['LIQ_CLUSTERS'] = 0  # Initialize if no futures data
            df['LIQ_SIDE_RATIO'] = 0
            df['LIQ_VOLUME'] = 0
            
            # Calculate impact with zero division protection
            df['LIQUIDATION_IMPACT'] = df['LIQ_VOLUME'] / df['volume'].replace(0, 1e-6)
            if 'open_interest' in df:
                df['OI_MOMENTUM'] = df['open_interest'].pct_change().rolling(8).mean()
            if 'funding_rate' in df:    
                df['FUNDING_TREND'] = df['funding_rate'].diff().rolling(4).sum()
        df['EMA_20'] = pd.to_numeric(TA.EMA(df, 20), errors='coerce')
        df['EMA_50'] = pd.to_numeric(TA.EMA(df, 50), errors='coerce')
        df['VWAP'] = pd.to_numeric(TA.VWAP(df), errors='coerce')
        df['OBV'] = pd.to_numeric(TA.OBV(df), errors='coerce').ffill()
        df['ADX'] = pd.to_numeric(TA.ADX(df), errors='coerce')
        
        # Bollinger Bands with type conversion
        bb_upper, bb_mid, bb_lower = TA.BBANDS(df)
        df['BB_UPPER'] = pd.to_numeric(bb_upper, errors='coerce')
        df['BB_MIDDLE'] = pd.to_numeric(bb_mid, errors='coerce')
        df['BB_LOWER'] = pd.to_numeric(bb_lower, errors='coerce')
        
        # Divergence detection
        df['RSI_14'] = TA.RSI(df, 14)
        df['PRICE_DELTA'] = df['close'].diff()
        df['RSI_DELTA'] = df['RSI_14'].diff()
        df['BULLISH_DIVERGENCE'] = (df['PRICE_DELTA'] < 0) & (df['RSI_DELTA'] > 0)
        df['BEARISH_DIVERGENCE'] = (df['PRICE_DELTA'] > 0) & (df['RSI_DELTA'] < 0)
        
        # Enhanced MACD analysis
        macd = TA.MACD(df)
        df['MACD'] = macd['MACD']
        df['SIGNAL'] = macd['SIGNAL']
        df['HISTOGRAM'] = (macd['MACD'] - macd['SIGNAL']).rolling(3).mean()

        # Volatility metrics
        df['ATR'] = TA.ATR(df, 14)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
        df['VWAP_VOLATILITY'] = df['VWAP'] * df['volume'].rolling(20).std()

        # Volume Profile Analysis (VPOC)
        price_range = df['close'].max() - df['close'].min()
        bins = int(price_range / (df['close'].median() * 0.001)) or 100  # Dynamic bin sizing
        df['price_bins'] = pd.cut(df['close'], bins=bins)
        volume_profile = df.groupby('price_bins', observed=False)['volume'].sum()
        
        if not volume_profile.empty:
            vpoc_bin = volume_profile.idxmax()
            df['VPOC_LEVEL'] = vpoc_bin.mid
            df['VPOC_STRENGTH'] = volume_profile.max() / volume_profile.sum()
        else:
            df['VPOC_LEVEL'] = df['VWAP']
            df['VPOC_STRENGTH'] = 0
            
        df['VPOC_DELTA'] = df['VPOC_LEVEL'].diff().rolling(4).mean()
        
        # Integrate with liquidation clusters
        df['VPOC_LIQ_RATIO'] = df['LIQ_CLUSTERS'] / (df['VPOC_STRENGTH'] + 1e-6)
        
        latest = df.iloc[-1]
        # Add these to analyze_data() function before returning:
        latest = df.iloc[-1]

        # Add bearish signals
        bearish_signals = sum([
            latest['BEARISH_DIVERGENCE'],
            latest['price'] < latest['BB_LOWER'],
            latest['EMA_20'] < latest['EMA_50'],
            latest['RSI'] > 70,
            latest['MACD'] < 0
        ])

        # Add bullish signals 
        bullish_signals = sum([
            latest['BULLISH_DIVERGENCE'],
            latest['price'] > latest['BB_UPPER'],
            latest['EMA_20'] > latest['EMA_50'],
            latest['RSI'] < 30,
            latest['MACD'] > 0
        ])

        return {
            'price': latest['close'],
            'rsi': float(latest['RSI']),
            'ema': float(latest['EMA_20']),
            'ema50': float(latest['EMA_50']),
            'macd': float(latest['HISTOGRAM']),
            'vwap': float(latest['VWAP']),
            'obv_trend': "↑" if float(latest['OBV']) > float(df['OBV'].iloc[-5]) else "↓",
            'adx': float(latest['ADX']),
            'bb_upper': float(latest['BB_UPPER']),
            'bb_lower': float(latest['BB_LOWER']),
            'atr': float(latest['ATR']),
            'bb_width': float(latest['BB_WIDTH']),
            'vwap_vol': float(latest['VWAP_VOLATILITY']),
            'liq_impact': float(latest.get('LIQUIDATION_IMPACT', 0.0)),
            'liq_side_ratio': float(latest.get('LIQ_SIDE_RATIO', 0.0)),
            'vpoc_level': float(latest.get('VPOC_LEVEL', 0.0)),
            'vpoc_strength': float(latest.get('VPOC_STRENGTH', 0.0)),
            'vpoc_delta': float(latest.get('VPOC_DELTA', 0.0)),
            'vpoc_liq_ratio': float(latest.get('VPOC_LIQ_RATIO', 0.0)),  # Added comma here
            'quant_confidence': 0.0,  # Calculated in bot.py
            'bullish_score': bullish_signals,
            'bearish_score': bearish_signals,
            'trend_direction': "bullish" if bullish_signals > bearish_signals else "bearish"
        }
    except Exception as e:
        return {'error': f"Analysis error: {str(e)}"}
