from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import data_fetcher
import analysis
import gemini_processor
from binance.exceptions import BinanceAPIException
from ml_model import MLModel
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the ML Model
ml_model = MLModel()

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
            
            # Get machine learning prediction confidence
            ml_confidence = ml_model.predict(df)
            ta['ml_confidence'] = 100 if ml_confidence == 1 else 0  # 100 for Bullish, 0 for Bearish
            
            timeframe_data[tf] = ta

        # Generate consolidated analysis
        analysis_text = f"📊 *{raw_symbol} Multi-Timeframe Analysis*\n\n"
        analysis_text += "```\n"
        analysis_text += "TF    | Price    | RSI  | EMA20/50   | MACD     | ADX  | BB Position | Q-Conf | ML Conf\n"
        analysis_text += "-------------------------------------------------------------------------------\n"
        
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
                f"| {ta['ml_confidence']:>5.1f}%\n"
            )
            
        analysis_text += "```\n\n"
        
        # Final message
        final_message = f"""
📈 Final Analysis for {raw_symbol}:
{analysis_text}

📊 Quantitative Confidence Scores:
5m: {timeframe_data['5m']['quant_confidence']}%
1h: {timeframe_data['1h']['quant_confidence']}%
1d: {timeframe_data['1d']['quant_confidence']}%

⚠️ Disclaimer: This is not financial advice. Always do your own research.
        """
        
        await update.message.reply_text(final_message)

    except BinanceAPIException as e:
        await update.message.reply_text(f"❌ Binance API Error: {e.message}")
    except Exception as e:
        await update.message.reply_text(f"❌ Unexpected error: {str(e)}")
