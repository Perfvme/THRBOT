# bot.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import config
import data_fetcher
import analysis
import gemini_processor
import ml_model
import time
import numpy as np

# Initialize ML engine
ml_engine = ml_model.MLEngine()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""🚀 Smart Trading Bot
/analyze <coin> - Get analysis (e.g. /analyze BTC)

Features:
- Scalping (5-15m) & Swing (1-4h) Signals
- Risk-Managed Entries (1:1.5-2 RR)
- Free API Compatible
- VPS Optimized""")

async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("❌ Please provide a coin symbol")
            return

        symbol = context.args[0].upper()
        timeframes = ['5m', '15m', '1h', '4h']
        
        # Get data with rate limiting
        analysis_data = {}
        for tf in timeframes:
            df, error = data_fetcher.get_crypto_data(symbol, tf)
            if error:
                await update.message.reply_text(f"❌ {error}")
                return
                
            result = analysis.analyze_data(df, symbol)
            if 'error' in result:
                await update.message.reply_text(f"❌ Analysis failed: {result['error']}")
                return
            
            analysis_data[tf] = result
            time.sleep(1)  # Rate limit

        # Generate recommendations
        scalping = generate_scalping_signal(analysis_data)
        swing = generate_swing_signal(analysis_data)
        
        # Build message
        msg = f"📈 *{symbol} Trading Signals*\n\n"
        msg += f"🔷 *Scalping (5-15m)*\n"
        msg += f"Entry: ${scalping['entry']:.2f}\n"
        msg += f"Targets: ${scalping['tp1']:.2f} → ${scalping['tp2']:.2f}\n"
        msg += f"Stop Loss: ${scalping['sl']:.2f} (-{scalping['risk_pct']:.1f}%)\n"
        msg += f"Risk/Reward: 1:{scalping['rr']:.1f}\n\n"

        msg += f"🌙 *Swing (1-4h)*\n"
        msg += f"Entry Zone: ${swing['entry_low']:.2f}-{swing['entry_high']:.2f}\n"
        msg += f"Targets: ${swing['tp1']:.2f} → ${swing['tp2']:.2f}\n"
        msg += f"Stop Loss: ${swing['sl']:.2f} (-{swing['risk_pct']:.1f}%)\n"
        msg += f"Risk/Reward: 1:{swing['rr']:.1f}\n\n"

        # Add ML confidence
        ml_pred = ml_engine.predict(df)
        msg += f"🤖 ML Confidence: {ml_pred['confidence']:.1f}%\n"
        
        # Add AI analysis
        await update.message.reply_text(msg + "\n🔄 Generating AI Insights...")
        ai_analysis = gemini_processor.get_gemini_analysis(analysis_data)
        await update.message.reply_text(f"💡 *AI Recommendations:*\n{ai_analysis}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

def generate_scalping_signal(data):
    """Generate scalping signals with 1:1.5 RR"""
    tf = data['5m']
    atr = tf['atr']
    price = tf['price']
    
    entry = price + (atr * 0.3)
    sl = price - (atr * 0.5)
    tp1 = entry + (entry - sl) * 1.5
    tp2 = tp1 + (atr * 0.5)
    
    return {
        'entry': round(entry, 2),
        'sl': round(sl, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2),
        'risk_pct': abs((sl - price)/price * 100),
        'rr': 1.5
    }

def generate_swing_signal(data):
    """Generate swing signals with 1:2 RR"""
    tf = data['4h']
    atr = tf['atr']
    price = tf['price']
    
    entry_low = price - (atr * 0.2)
    entry_high = price + (atr * 0.2)
    sl = price - (atr * 1.2)
    tp1 = price + (atr * 2.4)
    tp2 = tp1 + (atr * 1.0)
    
    return {
        'entry_low': round(entry_low, 2),
        'entry_high': round(entry_high, 2),
        'sl': round(sl, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2),
        'risk_pct': abs((sl - price)/price * 100),
        'rr': 2.0
    }

if __name__ == '__main__':
    # Initial ML training
    ml_engine.train_initial_model()
    
    # Start bot
    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_coin))
    print("🤖 Trading Bot Running...")
    app.run_polling()