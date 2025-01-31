import multiprocessing
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import config
import data_fetcher
import analysis
import gemini_processor
from binance.exceptions import BinanceAPIException
from apscheduler.schedulers.background import BackgroundScheduler
from ml_model import ml
import textwrap
import time
import traceback
import logging
import numpy as np

# Configure logging first
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def start_scheduler():
    """Initialize scheduler without multiprocessing"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(ml.train, 'interval', hours=24)
    scheduler.start()
    logger.info("Scheduler started")
    return scheduler

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(textwrap.dedent("""\
            🚀 Crypto Analysis Bot 3.0 🚀
            
            /analyze <coin> - Get analysis for a cryptocurrency
            Example: /analyze BTC"""))
    except Exception as e:
        logger.error(f"Start error: {str(e)}")

async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("❌ Please provide a coin symbol")
            return

        symbol = context.args[0].upper().strip()
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analysis_data = {}

        # Simple data collection
        for tf in timeframes:
            try:
                df, error = data_fetcher.get_crypto_data(symbol, tf)
                if error: continue
                
                ta = analysis.analyze_data(df, symbol)
                if 'error' in ta: continue

                # ML prediction
                ml_pred = ml.predict({
                    'RSI': ta['rsi'],
                    'EMA_20': ta.get('ema', ta.get('price', 0)),
                    'EMA_50': ta.get('ema50', ta.get('price', 0)),
                    'MACD': ta['macd'],
                    'VWAP': ta['vwap'],
                    'ADX': ta['adx'],
                    'funding_rate': ta.get('funding_rate', 0),
                    'open_interest': ta.get('open_interest', 0),
                    'LIQUIDATION_IMPACT': ta.get('liq_impact', 0)
                })
                
                analysis_data[tf] = {
                    'price': ta['price'],
                    'ml_confidence': ml_pred['confidence'],
                    'quant_confidence': min(100, max(0, 0.6*ta['quant_confidence'] + 0.4*ml_pred['confidence']))
                }
                
            except Exception as e:
                logger.error(f"{tf} error: {str(e)}")
                continue

        # Generate output
        output = f"📊 *{symbol} Analysis*\n\n"
        output += "```\nTF    | Price    | ML Conf | Q-Conf\n"
        output += "---------------------------------------\n"
        
        for tf in timeframes:
            data = analysis_data.get(tf, {})
            output += (
                f"{tf.upper().ljust(4)} "
                f"| ${data.get('price',0):>7.2f} "
                f"| {data.get('ml_confidence',50):>6.1f}% "
                f"| {data.get('quant_confidence',50):>5.1f}%\n"
            )
        output += "```\n"

        await update.message.reply_text(output)

    except Exception as e:
        await update.message.reply_text("🔴 Temporary system issue")
        logger.error(f"Analysis error: {str(e)}")

def main():
    """Simplified main function"""
    scheduler = start_scheduler()
    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_coin))
    
    try:
        logger.info("🤖 Bot started")
        app.run_polling()
    finally:
        scheduler.shutdown()
        logger.info("🛑 Clean shutdown")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
