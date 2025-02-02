# bot.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import data_fetcher
import analysis
import gemini_processor
import ml_model
import threading
import time
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging

# Load environment variables from the .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ml_engine = ml_model.CryptoML()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with instructions"""
    await update.message.reply_text("""🚀 Crypto Analysis Bot 🚀

Commands:
/analyze <coin> - Analyze cryptocurrency
/mlstatus - Show machine learning system status

Features:
- Quantitative + ML confidence scores
- Multi-timeframe analysis
- AI-powered recommendations
- Real-time system monitoring""")

async def ml_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Report ML system status"""
    try:
        status_text = "🤖 Machine Learning System Status\n\n"
        
        # Get training data stats
        data_counts = ml_engine.get_data_counts()
        status_text += f"📊 Training Data:\n"
        status_text += f"- 5m timeframe: {data_counts.get('5m', 0)} samples\n"
        status_text += f"- 1h timeframe: {data_counts.get('1h', 0)} samples\n"
        
        # Get model info
        model_info = ml_engine.get_model_info()
        status_text += "\n🧠 Model Performance:\n"
        for tf in ['5m', '1h']:
            if tf in model_info:
                status_text += (
                    f"- {tf} model: {model_info[tf]['accuracy']:.1f}% accuracy\n"
                    f"  Last trained: {model_info[tf]['last_trained']}\n"
                )
        
        if not model_info:
            status_text += "\n⚠️ Models not trained yet (collecting data)"
        
        status_text += "\n⚙️ System Status: "
        status_text += "Operational" if data_counts.get('5m', 0) > 100 else "Initializing"
        
        await update.message.reply_text(status_text)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Status check failed: {str(e)}")

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
            
            # Get the trend strength for bullish and bearish signals
            bullish_signals = ta['bullish_score']
            bearish_signals = ta['bearish_score']
            
            # Determine the trend direction and set a multiplier for confidence scaling
            if bullish_signals > bearish_signals:
                trend_score = 1.0
                direction_multiplier = 1.0
            elif bearish_signals > bullish_signals:
                trend_score = 1.0
                direction_multiplier = -1.0
            else:
                trend_score = 0.5
                direction_multiplier = 0.0

            # Calculate original quantitative confidence
            adx_score = min(ta['adx']/60, 1) if ta['adx'] else 0
            rsi_score = 1 - abs(ta['rsi'] - 50) / 50 if ta['rsi'] else 0.5
            quant_confidence = ((adx_score * 0.3) + (rsi_score * 0.2) + (trend_score * 0.5)) * abs(direction_multiplier) * 100
            ta['quant_confidence'] = max(-100, min(round(quant_confidence, 1), 100))
            
            # Add ML confidence
            current_features = ta['ml_features']
            current_features['timeframe'] = tf
            ml_confidence = ml_engine.predict_confidence({
                'rsi': ta['rsi'],
                'ema20': ta['ema'],
                'ema50': ta['ema50'],
                'macd': ta['macd'],
                'adx': ta['adx'],
                'bb_width': ta['bb_width'],
                'liq_impact': ta['liq_impact']
            }, tf)
            ta['ml_confidence'] = ml_confidence
            
            # Save features (delayed update for returns)
            if timeframe_data.get(tf):
                prev = timeframe_data[tf]
                time_diff = (datetime.now() - datetime.fromtimestamp(prev['ml_features']['timestamp']/1000))
                
                if tf == '5m' and time_diff > timedelta(minutes=5):
                    prev['ml_features']['next_5m_return'] = (ta['price'] - prev['price']) / prev['price']
                elif tf == '1h' and time_diff > timedelta(hours=1):
                    prev['ml_features']['next_1h_return'] = (ta['price'] - prev['price']) / prev['price']
                
                ml_engine.save_features(prev['ml_features'])
            
            timeframe_data[tf] = ta

        # Generate consolidated analysis
        analysis_text = f"📊 *{raw_symbol} Multi-Timeframe Analysis*\n\n"
        analysis_text += "```\n"
        analysis_text += "TF    | Price    | RSI  | EMA20/50   | MACD     | ADX  | BB Position\n"
        analysis_text += "---------------------------------------------------------------------\n"
        
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
                f"| {bb_position.ljust(6)}\n"
            )
            
        analysis_text += "```\n\n"
        analysis_text += f"🔑 Key Levels:\n"
        analysis_text += f"• Strong Support: ${min([ta['ema50'] for ta in timeframe_data.values()]):.2f}\n"
        analysis_text += f"• Strong Resistance: ${max([ta['ema50'] for ta in timeframe_data.values()]):.2f}\n\n"

        # Trend alignment analysis
        trend_strength = {'bullish': 0, 'bearish': 0}
        for tf in timeframes:
            ta = timeframe_data[tf]
            if ta['trend_direction'] == "bullish":
                trend_strength['bullish'] += 1
            else:
                trend_strength['bearish'] += 1
        
        analysis_text += f"🔀 Trend Consensus: Bullish {trend_strength['bullish']}/5 vs Bearish {trend_strength['bearish']}/5\n"

        # Generate AI recommendations
        await update.message.reply_text("🔄 Generating AI recommendations...")
        recommendations = gemini_processor.get_gemini_analysis(analysis_text)
        
        # Format final message
        final_message = f"""
📈 Final Analysis for {raw_symbol}:
{recommendations if "⚠️" not in recommendations else "⚠️ Partial Analysis (Verify Manually):\n" + recommendations}

📊 Confidence Scores:
│           │ Quantitative │ ML Model │
├───────────┼──────────────┼──────────┤
│ 5m       │ {timeframe_data['5m']['quant_confidence']:>5.1f}%     │ {timeframe_data['5m']['ml_confidence']:>5.1f}% │
│ 1h       │ {timeframe_data['1h']['quant_confidence']:>5.1f}%     │ {timeframe_data['1h']['ml_confidence']:>5.1f}% │ 
│ 1d       │ {timeframe_data['1d']['quant_confidence']:>5.1f}%     │ {'N/A':^8} │

⚠️ Disclaimer: This is not financial advice.
"""        
        await update.message.reply_text(final_message)

    except BinanceAPIException as e:
        await update.message.reply_text(f"❌ Binance API Error: {e.message}")
    except Exception as e:
        await update.message.reply_text(f"❌ Unexpected error: {str(e)}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_coin))
    app.add_handler(CommandHandler("mlstatus", ml_status))
    
    # Start background training scheduler
    def train_models():
        while True:
            try:
                for tf in ['5m', '1h']:
                    accuracy = ml_engine.train_model(tf)
                    print(f"Retrained {tf} model with accuracy: {accuracy:.2f}")
            except Exception as e:
                print(f"Training failed: {str(e)}")
            time.sleep(3600*6)  # Retrain every 6 hours

    training_thread = threading.Thread(target=train_models, daemon=True)
    training_thread.start()
    
    print("🤖 Bot is running... Press CTRL+C to stop")
    app.run_polling()
