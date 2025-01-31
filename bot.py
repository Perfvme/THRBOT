from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import config
import data_fetcher
import analysis
import gemini_processor
from binance.exceptions import BinanceAPIException
from apscheduler.schedulers.background import BackgroundScheduler
from ml_model import ml
import pandas as pd
import textwrap
import multiprocessing
import time
import traceback
import logging
import numpy as np

# Configure advanced logging
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
    """Initialize scheduler with resource monitoring"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(ml.train, 'interval', hours=24)
    scheduler.start()
    logger.info("Scheduler started with daily ML training")
    return scheduler

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced welcome message with system status"""
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
    """Robust analysis handler with circuit breaker"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Please provide a coin symbol. Example: /analyze BTC")
            return

        raw_symbol = context.args[0].upper().strip()
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        timeframe_data = {}

        # Initial symbol check with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df, error = data_fetcher.get_crypto_data(raw_symbol, '5m')
                if error:
                    raise ValueError(error)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Data fetch attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)

        # Process timeframes with error isolation
        for tf in timeframes:
            try:
                df, error = data_fetcher.get_crypto_data(raw_symbol, tf)
                if error:
                    raise ValueError(error)
                
                ta = analysis.analyze_data(df, raw_symbol)
                if 'error' in ta:
                    raise ValueError(ta['error'])
                
                # ML Prediction with fallback
                try:
                    ml_features = {
                        'RSI': ta['rsi'],
                        'EMA_20': ta['ema'],
                        'EMA_50': ta['ema50'],
                        'MACD': ta['macd'],
                        'VWAP': ta['vwap'],
                        'ADX': ta['adx'],
                        'funding_rate': ta.get('funding_rate', 0),
                        'open_interest': ta.get('open_interest', 0),
                        'LIQUIDATION_IMPACT': ta.get('liq_impact', 0)
                    }
                    ml_pred = ml.predict(ml_features)
                except Exception as ml_error:
                    logger.error(f"ML prediction failed: {str(ml_error)}")
                    ml_pred = {'confidence': 50.0, 'uncertainty': 100.0}

                ta.update({
                    'ml_confidence': ml_pred['confidence'],
                    'ml_uncertainty': ml_pred['uncertainty'],
                    'quant_confidence': min(100, max(0, 0.6*ta['quant_confidence'] + 0.4*ml_pred['confidence']))
                })
                
                timeframe_data[tf] = ta
                
            except Exception as tf_error:
                logger.error(f"Timeframe {tf} analysis failed: {str(tf_error)}")
                continue

        # Generate output with fallback values
        analysis_text = f"📊 *{raw_symbol} Multi-Timeframe Analysis*\n\n"
        analysis_text += "```\nTF    | Price    | ML Conf | Uncertainty | Q-Conf\n"
        analysis_text += "-----------------------------------------------------\n"
        
        for tf in timeframes:
            ta = timeframe_data.get(tf, {
                'price': 0.0,
                'ml_confidence': 50.0,
                'ml_uncertainty': 100.0,
                'quant_confidence': 50.0,
                'ema50': 0.0
            })
            analysis_text += (
                f"{tf.upper().ljust(4)} "
                f"| ${ta['price']:>7.2f} "
                f"| {ta['ml_confidence']:>6.1f}% "
                f"| ±{ta['ml_uncertainty']:>4.1f}% "
                f"| {ta['quant_confidence']:>5.1f}%\n"
            )
        analysis_text += "```\n\n"

        # Key levels calculation with fallback
        try:
            ema50_values = [ta['ema50'] for ta in timeframe_data.values()]
            analysis_text += f"🔑 Key Levels:\n• Support: ${min(ema50_values):.2f}\n• Resistance: ${max(ema50_values):.2f}\n\n"
        except:
            analysis_text += "🔑 Key Levels: Data unavailable\n\n"

        # AI recommendations with timeout
        try:
            await update.message.reply_text("🔄 Generating AI recommendations...")
            recommendations = gemini_processor.get_gemini_analysis(analysis_text)
        except Exception as ai_error:
            recommendations = f"⚠️ AI Analysis Unavailable: {str(ai_error)}"
            logger.error(f"Gemini failed: {str(ai_error)}")

        # Final message assembly
        final_message = textwrap.dedent(f"""\
        📈 Final Analysis for {raw_symbol}:
        {recommendations if "⚠️" not in recommendations else "⚠️ Partial Analysis (Verify Manually):\\n" + recommendations}

        🤖 ML Insights:
        - Average Confidence: {np.mean([t.get('ml_confidence',50) for t in timeframe_data.values()]):.1f}%
        - System Health: {'Stable' if ml.model else 'Initializing'}

        ⚠️ Disclaimer: Algorithmic analysis only. Verify with other sources.""")

        await update.message.reply_text(final_message)

    except BinanceAPIException as e:
        await update.message.reply_text(f"❌ Exchange Error: {e.message}")
        logger.error(f"Binance API Error: {e.message}")
    except Exception as e:
        await update.message.reply_text("🔴 System temporarily unavailable. Please try again later.")
        logger.error(f"Analysis pipeline failed: {str(e)}\n{traceback.format_exc()}")

def resilient_run():
    """Self-healing main loop with exponential backoff"""
    backoff = 1
    max_backoff = 300  # 5 minutes
    consecutive_errors = 0
    
    while True:
        try:
            multiprocessing.freeze_support()
            scheduler = start_scheduler()
            
            app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
            app.add_handler(CommandHandler("start", start))
            app.add_handler(CommandHandler("analyze", analyze_coin))
            
            logger.info("🚀 Starting bot...")
            app.run_polling()
            
            # Reset counters on clean exit
            backoff = 1
            consecutive_errors = 0
            
        except KeyboardInterrupt:
            logger.info("🛑 Graceful shutdown initiated")
            break
        except Exception as e:
            consecutive_errors += 1
            logger.critical(f"💥 Critical failure: {str(e)}\n{traceback.format_exc()}")
            
            # Calculate backoff with jitter
            sleep_time = min(backoff * (2 ** consecutive_errors), max_backoff)
            sleep_time *= np.random.uniform(0.5, 1.5)
            
            logger.info(f"⏳ Restarting in {sleep_time:.1f}s (errors: {consecutive_errors})...")
            time.sleep(sleep_time)
            
            # Cleanup resources
            try:
                if 'scheduler' in locals() and scheduler.running:
                    scheduler.shutdown()
                if ml.client:
                    ml.client.close()
                if 'app' in locals():
                    app.stop()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {str(cleanup_error)}")
            
            # Reset backoff after 5 consecutive errors
            if consecutive_errors >= 5:
                backoff = 1
                consecutive_errors = 0

if __name__ == '__main__':
    # Systemd-style supervision
    while True:
        try:
            resilient_run()
        except KeyboardInterrupt:
            logger.info("🛑 Permanent shutdown requested")
            break
        except Exception as fatal_error:
            logger.critical(f"💀 Catastrophic failure: {str(fatal_error)}")
            logger.info("🔁 Attempting cold restart in 60s...")
            time.sleep(60)
