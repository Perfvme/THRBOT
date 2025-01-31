import multiprocessing
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import config
import data_fetcher
import analysis
import gemini_processor
from binance.exceptions import BinanceAPIException
from apscheduler.schedulers.background import BackgroundScheduler
from ml_model import MLSystem
import textwrap
import time
import traceback
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize ML system
ml = MLSystem()

def start_scheduler():
    """Initialize scheduler without Dask dependencies"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(ml.train, 'interval', hours=24)
    scheduler.start()
    logger.info("Scheduler started with daily ML training")
    return scheduler

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        health_status = "🟢 Operational" if ml.model else "🟡 Initializing"
        await update.message.reply_text(textwrap.dedent(f"""\
            🚀 Crypto Analysis Bot 3.0 🚀
            
            System Status: {health_status}
            ML Model: {'Trained' if ml.model else 'Pending'}
            
            Usage:
            /analyze <coin>  Example: /analyze BTC"""))
    except Exception as e:
        logger.error(f"Start command error: {str(e)}")

async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("❌ Please provide a coin symbol. Example: /analyze BTC")
            return

        raw_symbol = context.args[0].upper().strip()
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        timeframe_data = {}

        # Data fetching with retries
        for attempt in range(3):
            try:
                df, error = data_fetcher.get_crypto_data(raw_symbol, '5m')
                if error: raise ValueError(error)
                break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2 ** attempt)

        # Process timeframes
        for tf in timeframes:
            try:
                df, error = data_fetcher.get_crypto_data(raw_symbol, tf)
                if error: continue
                
                ta = analysis.analyze_data(df, raw_symbol)
                if 'error' in ta: continue

                ml_pred = ml.predict({
                    'RSI': ta['rsi'],
                    'EMA_20': ta['ema'],
                    'EMA_50': ta['ema50'],
                    'MACD': ta['macd'],
                    'VWAP': ta['vwap'],
                    'ADX': ta['adx'],
                    'funding_rate': ta.get('funding_rate', 0),
                    'open_interest': ta.get('open_interest', 0),
                    'LIQUIDATION_IMPACT': ta.get('liq_impact', 0)
                })
                
                ta.update({
                    'ml_confidence': ml_pred['confidence'],
                    'ml_uncertainty': ml_pred['uncertainty'],
                    'quant_confidence': min(100, max(0, 0.6*ta['quant_confidence'] + 0.4*ml_pred['confidence']))
                })
                
                timeframe_data[tf] = ta
                
            except Exception as tf_error:
                logger.error(f"Timeframe {tf} error: {str(tf_error)}")
                continue

        # Generate analysis output
        analysis_text = f"📊 *{raw_symbol} Analysis*\n\n"
        analysis_text += "```\nTF    | Price    | ML Conf | Uncertainty | Q-Conf\n"
        analysis_text += "-----------------------------------------------------\n"
        
        for tf in timeframes:
            ta = timeframe_data.get(tf, {})
            analysis_text += (
                f"{tf.upper().ljust(4)} "
                f"| ${ta.get('price',0):>7.2f} "
                f"| {ta.get('ml_confidence',50):>6.1f}% "
                f"| ±{ta.get('ml_uncertainty',100):>4.1f}% "
                f"| {ta.get('quant_confidence',50):>5.1f}%\n"
            )
        analysis_text += "```\n\n"

        # Generate recommendations
        try:
            await update.message.reply_text("🔄 Generating AI recommendations...")
            recommendations = gemini_processor.get_gemini_analysis(analysis_text)
        except Exception as e:
            recommendations = f"⚠️ AI Analysis Unavailable: {str(e)}"

        # Final message
        final_message = textwrap.dedent(f"""\
        📈 Final Analysis for {raw_symbol}:
        {recommendations}

        🤖 ML Insights:
        - Average Confidence: {np.mean([t.get('ml_confidence',50) for t in timeframe_data.values()]):.1f}%
        - System Health: {'Stable' if ml.model else 'Initializing'}

        ⚠️ Disclaimer: Algorithmic analysis only. Verify with other sources.""")

        await update.message.reply_text(final_message)

    except BinanceAPIException as e:
        await update.message.reply_text(f"❌ Exchange Error: {e.message}")
    except Exception as e:
        await update.message.reply_text("🔴 Temporary system issue. Please try again.")
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")

def main():
    """Main execution loop"""
    scheduler = start_scheduler()
    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_coin))
    
    try:
        logger.info("🤖 Bot starting...")
        app.run_polling()
    except KeyboardInterrupt:
        logger.info("🛑 Graceful shutdown")
    finally:
        scheduler.shutdown()
        logger.info("🧹 Cleanup completed")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
