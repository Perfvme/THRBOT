import google.generativeai as genai
import textwrap
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import os
import logging
import re

# Load environment variables from the .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([TELEGRAM_TOKEN, BINANCE_API_KEY, BINANCE_SECRET_KEY, GEMINI_API_KEY]):
    logging.error("One or more environment variables are missing. Check your .env file.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def validate_response(text):
    """Flexible validation with key section checks"""
    required_keywords = [
        "ENTRY", "TARGETS", "STOP LOSS", 
        "VOLATILITY", "CONFIDENCE", "RISK"
    ]
    return sum(kw in text.upper() for kw in required_keywords) >= 3

# Modify the validate_recommendations function
def validate_recommendations(text, current_price):
    """Ensure numerical validity of generated recommendations"""
    numbers = [float(x) for x in re.findall(r'\d+\.?\d*', text)] if text else []
    valid = True
    details = ""
    
    try:
        if len(numbers) >= 5:
            entry_low = numbers[0] if len(numbers) > 0 else 0.0
            entry_high = numbers[1] if len(numbers) > 1 else 0.0
            tp1 = numbers[2] if len(numbers) > 2 else 0.0
            tp2 = numbers[3] if len(numbers) > 3 else 0.0
            sl = numbers[4] if len(numbers) > 4 else 0.0
            
            risk = abs((sl - current_price)/current_price*100)
            reward = abs((tp1 - current_price)/current_price*100)
            rrr = reward/risk if risk != 0 else 0
            
            details = f"""
ENTRY: {entry_low:.2f}-{entry_high:.2f}
TP1: {tp1:.2f} ({reward:.1f}%)
TP2: {tp2:.2f} 
SL: {sl:.2f} (-{risk:.1f}%)
RRR: 1:{rrr:.1f}
"""
    except:
        valid = False
    
    return {
        'is_valid': valid,
        'sanitized': details if valid else "⚠️ Invalid levels detected - verify manually"
    }

def get_gemini_analysis(prompt, ta_data):
    """Generate analysis with structured prompting"""
    try:
        structured_prompt = f"""
CRYPTO ANALYSIS REQUEST - RESPONSE TEMPLATE:

CURRENT PRICE: ${ta_data['price']:.2f}
KEY TECHNICALS:
- RSI: {ta_data['rsi']:.1f}
- EMA20/50: {ta_data['ema']:.2f}/{ta_data['ema50']:.2f}
- ADX: {ta_data['adx']:.1f} ({'Strong' if ta_data['adx'] > 25 else 'Weak'} Trend)
- Volatility (ATR): ${ta_data['atr']:.2f}

KEY LEVELS:
- Support: ${ta_data['swing_low']:.2f}
- Resistance: ${ta_data['swing_high']:.2f}
- Fibonacci: {ta_data['fib_236']:.2f} | {ta_data['fib_500']:.2f} | {ta_data['fib_618']:.2f}
- VWAP Deviations: ±2σ = {ta_data['vwap_dev2']:.2f}/{ta_data['vwap_dev-2']:.2f}

ML INSIGHTS:
- Direction Confidence: {ta_data['ml_confidence']:.1f}%
- Uncertainty Factor: {ta_data['ml_uncertainty']:.1f}%
- Suggested Range: ±${ta_data['suggested_width']:.2f}

GENERATE:
1. Entry zone considering volatility and key levels
2. Two TP targets aligned with Fibonacci/VWAP
3. SL placement beyond swing points
4. Risk/Reward ratio between 1:2-1:3

STRICT FORMAT:
ENTRY: [price]-[price]
TARGETS: [price] → [price]  
STOP LOSS: [price] (-X.X%)
"""
        response = model.generate_content(
            textwrap.shorten(structured_prompt, width=30000, placeholder="... [truncated]")
        )
        
        if validate_response(response.text):
            return response.text
        else:
            # Attempt format correction
            corrected = response.text.replace("->", "→").replace("SL:", "STOP LOSS:")
            if validate_response(corrected):
                return corrected
            return "⚠️ AI generated incomplete analysis. Verify technicals manually."
            
    except google_exceptions.InvalidArgument as e:
        return f"❌ Error: Analysis too complex ({e.message})"
    except Exception as e:
        return f"❌ API Error: {str(e)}"
