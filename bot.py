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

# Initialize scheduler for daily ML training
scheduler = BackgroundScheduler()
scheduler.add_job(ml.train, 'interval', hours=24)
scheduler.start()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with updated instructions"""
    await update.message.reply_text(textwrap.dedent("""\
        🚀 Crypto Analysis Bot 2.0 🚀
        
        Usage:
        /analyze <coin>  Example: /analyze BTC
        
        New Features:
        - ML-Powered Confidence Scores
        - Uncertainty Estimation
        - Adaptive Market Regime Detection
        - Risk Cluster Alerts
        - Memory-Efficient Analysis"""))

async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle analysis requests with ML integration"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Please provide a coin symbol. Example: /analyze BTC")
            return

        raw_symbol = context.args[0].upper().strip()
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        timeframe_data = {}

        # Initial symbol check
        df, error = data_fetcher.get_crypto_data(raw_symbol, '5m')
        if error:
            await update.message.reply_text(f"❌ {error}")
            return

        # Process all timeframes
        for tf in timeframes:
            df, error = data_fetcher.get_crypto_data(raw_symbol, tf)
            if error:
                await update.message.reply_text(f"❌ {tf} error: {error}")
                return
                
            ta = analysis.analyze_data(df, raw_symbol)
            if 'error' in ta:
                await update.message.reply_text(f"❌ Analysis failed: {ta['error']}")
                return
            
            # Get ML prediction
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
            
            # Update TA with ML results
            ta['ml_confidence'] = ml_pred['confidence']
            ta['ml_uncertainty'] = ml_pred['uncertainty']
            ta['quant_confidence'] = min(100, max(0, 0.6*ta['quant_confidence'] + 0.4*ta['ml_confidence']))
            
            timeframe_data[tf] = ta

        # Generate analysis table
        analysis_text = f"📊 *{raw_symbol} Multi-Timeframe Analysis*\n\n"
        analysis_text += "```\n"
        analysis_text += "TF    | Price    | ML Conf | Uncertainty | Q-Conf\n"
        analysis_text += "-----------------------------------------------------\n"
        
        for tf in timeframes:
            ta = timeframe_data[tf]
            analysis_text += (
                f"{tf.upper().ljust(4)} "
                f"| ${ta['price']:>7.2f} "
                f"| {ta['ml_confidence']:>6.1f}% "
                f"| ±{ta['ml_uncertainty']:>4.1f}% "
                f"| {ta['quant_confidence']:>5.1f}%\n"
            )
        analysis_text += "```\n\n"

        # Key levels calculation
        ema50_values = [ta['ema50'] for ta in timeframe_data.values()]
        analysis_text += f"🔑 Key Levels:\n"
        analysis_text += f"• Strong Support: ${min(ema50_values):.2f}\n"
        analysis_text += f"• Strong Resistance: ${max(ema50_values):.2f}\n\n"

        # Generate AI recommendations
        await update.message.reply_text("🔄 Generating AI recommendations...")
        try:
            recommendations = gemini_processor.get_gemini_analysis(analysis_text)
        except Exception as e:
            recommendations = f"⚠️ AI Analysis Unavailable: {str(e)}"

        # Format final message
        final_message = textwrap.dedent(f"""\
        📈 Final Analysis for {raw_symbol}:
        {recommendations if "⚠️" not in recommendations else "⚠️ Partial Analysis (Verify Manually):\\n" + recommendations}

        🤖 ML Insights:
        - Average Confidence: {sum(ta['ml_confidence'] for ta in timeframe_data.values())/5:.1f}%
        - Lowest Uncertainty: {min(ta['ml_uncertainty'] for ta in timeframe_data.values()):.1f}%
        - Risk Alert: {'High' if any(ta['liq_impact'] > 0.5 for ta in timeframe_data.values()) else 'Low'} Liquidation Risk

        📊 Confidence Overview:
        5m: Q-Conf {timeframe_data['5m']['quant_confidence']:.1f}% | ML {timeframe_data['5m']['ml_confidence']:.1f}%
        1h: Q-Conf {timeframe_data['1h']['quant_confidence']:.1f}% | ML {timeframe_data['1h']['ml_confidence']:.1f}%
        1d: Q-Conf {timeframe_data['1d']['quant_confidence']:.1f}% | ML {timeframe_data['1d']['ml_confidence']:.1f}%

        ⚠️ Disclaimer: Algorithmic analysis only. Verify with other sources.""")
        
        await update.message.reply_text(final_message)

    except BinanceAPIException as e:
        await update.message.reply_text(f"❌ Binance API Error: {e.message}")
    except Exception as e:
        await update.message.reply_text(f"❌ System Error: {str(e)}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_coin))
    print("🤖 ML-Powered Bot is running... Press CTRL+C to stop")
    app.run_polling()