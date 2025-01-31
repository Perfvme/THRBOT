# bot.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import data_fetcher
import analysis
import gemini_processor
from binance.exceptions import BinanceAPIException
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import ml_trainer
from dotenv import load_dotenv
import os
import logging
import threading

# Load environment variables from the .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([TELEGRAM_TOKEN, BINANCE_API_KEY, BINANCE_SECRET_KEY, GEMINI_API_KEY]):
    logging.error("One or more environment variables are missing. Check your .env file.")
    exit(1)

# Initialize ML scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(ml_trainer.ml_engine.train, 'cron', hour=3)  # Daily training at 3 AM
scheduler.add_job(ml_trainer.ml_engine.update_outcomes, 'interval', hours=1)  # Update outcomes hourly
scheduler.start()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with instructions"""
    await update.message.reply_text("""🚀 Crypto Analysis Bot 🚀

Usage:
/analyze <coin>  Example: /analyze BTC

Features:
- 5min to 1D timeframe analysis
- Technical indicators (RSI, EMA, MACD, ADX)
- AI-powered recommendations with risk management
- Support/resistance levels
- Quantitative & ML confidence scoring""")

async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle analysis requests"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Please provide a coin symbol. Example: /analyze BTC")
            return

        raw_symbol = context.args[0].upper().strip()
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        
        # Initial symbol check
        df, error = data_fetcher.get_crypto_data(raw_symbol, '5m')
        if error:
            await update.message.reply_text(f"❌ {error}")
            return

        timeframe_data = {}
        
        # Collect all timeframe data first
        for tf in timeframes:
            df, error = data_fetcher.get_crypto_data(raw_symbol, tf)
            if error:
                await update.message.reply_text(f"❌ {tf} error: {error}")
                return
                
            ta = analysis.analyze_data(df, raw_symbol)
            if 'error' in ta:
                await update.message.reply_text(f"❌ Analysis failed: {ta['error']}")
                return
            
            # Calculate quantitative confidence
            adx_score = min(ta['adx']/60, 1) if ta['adx'] else 0
            rsi_score = 1 - abs(ta['rsi']-50)/50 if ta['rsi'] else 0.5
            trend_score = 0.5  # Neutral baseline
            
            if (ta['price'] > ta['ema'] and ta['macd'] > 0 and ta['obv_trend'] == "↑"):
                trend_score = 1.0
            elif (ta['price'] < ta['ema'] and ta['macd'] < 0 and ta['obv_trend'] == "↓"):
                trend_score = 1.0
                
            quant_confidence = ((adx_score * 0.3) + (rsi_score * 0.2) + (trend_score * 0.5)) * 100
            ta['quant_confidence'] = max(0, min(round(quant_confidence, 1), 100))
            
            timeframe_data[tf] = ta

            # Save data for ML training
            ml_trainer.ml_engine.save_analysis({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': raw_symbol,
                'timeframe': tf,
                'rsi': ta['rsi'],
                'ema20': ta['ema'],
                'ema50': ta['ema50'],
                'macd': ta['macd'],
                'adx': ta['adx'],
                'bb_width': ta['bb_width'],
                'price': ta['price'],
                'trend_bullish': 1 if (ta['price'] > ta['ema'] and ta['macd'] > 0 and ta['obv_trend'] == "↑") else 0,
                'trend_bearish': 1 if (ta['price'] < ta['ema'] and ta['macd'] < 0 and ta['obv_trend'] == "↓") else 0,
                'outcome': None
            })

        # Generate ML predictions
        ml_confidences = {}
        for tf in timeframes:
            features = [
                timeframe_data[tf]['rsi'],
                timeframe_data[tf]['ema'],
                timeframe_data[tf]['ema50'],
                timeframe_data[tf]['macd'],
                timeframe_data[tf]['adx'],
                timeframe_data[tf]['bb_width'],
                timeframe_data[tf]['price']
            ]
            ml_confidences[tf] = ml_trainer.ml_engine.predict(features)

        # Generate consolidated analysis
        analysis_text = f"📊 *{raw_symbol} Multi-Timeframe Analysis*\n\n"
        analysis_text += "```\n"
        analysis_text += "TF    | Price    | RSI  | EMA20/50   | MACD     | ADX  | BB Position | Q-Conf | ML-Conf\n"
        analysis_text += "-----------------------------------------------------------------------------------------\n"
        
        for tf in timeframes:
            ta = timeframe_data[tf]
            bb_position = "Middle" if (ta['price'] > ta['bb_lower'] and ta['price'] < ta['bb_upper']) else ("Upper" if ta['price'] > ta['bb_upper'] else "Lower")
            analysis_text += (
                f"{tf.upper().ljust(4)} "
                f"| ${ta['price']:>7.2f} "
                f"| {ta['rsi']:>3.0f} "
                f"| {ta['ema']:>5.2f}/{ta['ema50']:>5.2f} "
                f"| {ta['macd']:>+7.4f} "
                f"| {ta['adx']:>3.0f} "
                f"| {bb_position.ljust(6)} "
                f"| {ta['quant_confidence']:>5.1f}% "
                f"| {ml_confidences[tf]:>5.1f}%\n"
            )
            
        analysis_text += "```\n\n"
        analysis_text += f"🔑 Key Levels:\n"
        analysis_text += f"• Strong Support: ${min([ta['ema50'] for ta in timeframe_data.values()]):.2f}\n"
        analysis_text += f"• Strong Resistance: ${max([ta['ema50'] for ta in timeframe_data.values()]):.2f}\n\n"

        # Trend alignment analysis
        trend_strength = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for tf in timeframes:
            ta = timeframe_data[tf]
            if (ta['price'] > ta['ema'] and ta['macd'] > 0 and ta['obv_trend'] == "↑"):
                trend_strength['bullish'] += 1
            elif (ta['price'] < ta['ema'] and ta['macd'] < 0 and ta['obv_trend'] == "↓"):
                trend_strength['bearish'] += 1
            else:
                trend_strength['neutral'] += 1
        
        analysis_text += f"🔀 Trend Consensus: Bullish {trend_strength['bullish']}/5, Bearish {trend_strength['bearish']}/5\n"

        # Generate recommendations
        await update.message.reply_text("🔄 Generating AI recommendations...")
        recommendations = gemini_processor.get_gemini_analysis(analysis_text)
        
        # Format final message
        final_message = f"""
📈 Final Analysis for {raw_symbol}:
{recommendations if "⚠️" not in recommendations else "⚠️ Partial Analysis (Verify Manually):\n" + recommendations}

📊 Confidence Scores (Quant/ML):
5m: {timeframe_data['5m']['quant_confidence']}% / {ml_confidences['5m']:.1f}%
1h: {timeframe_data['1h']['quant_confidence']}% / {ml_confidences['1h']:.1f}%
1d: {timeframe_data['1d']['quant_confidence']}% / {ml_confidences['1d']:.1f}%

⚠️ Disclaimer: This is not financial advice. Always do your own research.
        """
        
        await update.message.reply_text(final_message)

    except BinanceAPIException as e:
        await update.message.reply_text(f"❌ Binance API Error: {e.message}")
    except Exception as e:
        await update.message.reply_text(f"❌ Unexpected error: {str(e)}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_coin))
    print("🤖 Bot is running... Press CTRL+C to stop")
    app.run_polling()
