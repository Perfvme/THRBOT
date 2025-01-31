from binance.client import Client
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY, testnet=True)

def get_valid_symbols():
    """Get all active USDT pairs from Binance"""
    return [s['symbol'] for s in client.get_exchange_info()['symbols'] 
            if s['status'] == 'TRADING' and 'USDT' in s['symbol']]

def get_futures_data(symbol):
    """Get futures market data from Binance"""
    try:
        return {
            'funding_rate': float(client.futures_funding_rate(symbol=symbol)[-1]['fundingRate']),
            'open_interest': float(client.futures_open_interest(symbol=symbol)['openInterest']),
            'liquidations': client.futures_liquidation_orders(symbol=symbol, limit=10)
        }
    except:
        return None  # Fallback if not a futures contract

def get_crypto_data(symbol, timeframe):
    try:
        symbol = symbol.upper().replace(' ', '').replace('-', '')
        
        # Auto-complete USDT pair and check futures
        is_futures = symbol.endswith('PERP')
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        # Validate symbol
        valid_pairs = get_valid_symbols()
        if symbol not in valid_pairs:
            suggestions = [p for p in valid_pairs if p.startswith(symbol[:-4])]
            return None, f"Invalid symbol. Try: {', '.join(suggestions[:3])}"
        
        # Fetch data
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=300)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], None
        
    except Exception as e:
        return None, f"Binance Error: {str(e)}"
