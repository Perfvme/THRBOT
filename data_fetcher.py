from binance.client import Client
import pandas as pd
import config
from data_collector import DataCollector

client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY)
data_collector = DataCollector()

def get_futures_data(symbol):
    try:
        return {
            'funding_rate': float(client.futures_funding_rate(symbol=symbol)[-1]['fundingRate']),
            'open_interest': float(client.futures_open_interest(symbol=symbol)['openInterest']),
            'liquidations': client.futures_liquidation_orders(symbol=symbol, limit=5)
        }
    except:
        return None

def get_crypto_data(symbol, timeframe):
    try:
        symbol = symbol.upper().replace(' ', '').replace('-', '')
        if not symbol.endswith('USDT'):
            symbol += 'USDT'

        klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=100  # Reduced to stay under API limits
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]).apply(pd.to_numeric, errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        data_collector.store_data(df, symbol, timeframe)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], None
    except Exception as e:
        return None, f"Error: {str(e)}"