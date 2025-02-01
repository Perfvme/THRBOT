# gemini_processor.py
import google.generativeai as genai
import textwrap
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import os
import logging

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
    """Flexible validation with pattern matching"""
    patterns = [
        r"SCALP", 
        r"SWING", 
        r"PRICE ACTION",
        r"CONFIDENCE GUARDRAILS",
        r"Quantitative Confidence",
        r"AI Confidence"
    ]
    return all(re.search(p, text, re.IGNORECASE) for p in patterns)

def format_fallback_analysis(symbol, quant_confidence):
    """Fallback template with real quantitative values"""
    return f"""🔍 PARTIAL ANALYSIS REQUIRES VERIFICATION

🚀 SCALP (5-15m)
`Quantitative Confidence:` {quant_confidence['5m']}%
`AI Confidence:` 50.0% (Manual Check Needed)
`Key Levels:` 
- Support: ${quant_confidence['support']:.2f}
- Resistance: ${quant_confidence['resistance']:.2f}
`Action:` Verify RSI({quant_confidence['rsi']}) & MACD({quant_confidence['macd']:+})

🌙 SWING (1-4H)  
`Quantitative Confidence:` {quant_confidence['1h']}%
`Critical Levels:`
- POC: ${quant_confidence['poc']:.2f}
- Liquidation Zone: ${quant_confidence['liq_zone']:.2f}
`Check:` Volume Spike & Funding Rates

⚠️ MANUAL VERIFICATION STEPS:
1. Confirm Bollinger Band ({quant_confidence['bb_width']:.2f} width) alignment
2. Check 1D timeframe liquidity clusters
3. Verify exchange heatmap data"""

def sanitize_prompt(prompt):
    """Remove problematic characters/terms"""
    replacements = {
        "SHORT": "DOWNTREND",
        "LONG": "UPTREND",
        "STOP LOSS": "RISK LEVEL",
        "LIQUIDATION": "VOLUME CLUSTER"
    }
    for k, v in replacements.items():
        prompt = prompt.replace(k, v)
    return prompt

def get_gemini_analysis(prompt):
    try:
        # Clean and truncate prompt
        clean_prompt = sanitize_prompt(prompt)
        truncated_prompt = textwrap.shorten(clean_prompt, width=15000, placeholder="... [truncated]")
        
        # First attempt with full template
        response = model.generate_content(
            f"""STRICT FORMAT REQUIRED! ANALYZE THIS MARKET DATA:
{truncated_prompt}

MANDATORY SECTIONS:
1. 🚀 SCALP ANALYSIS
2. 🌙 SWING ANALYSIS  
3. 🔍 PRICE ACTION NOTES
4. ⚠️ RISK FACTORS

INCLUDE THESE METRICS IN EACH SECTION:
- Quantitative Confidence %
- AI Confidence %
- Key Levels
- Volume Analysis""",
            generation_config={"temperature": 0.4}
        )
        
        # First validation pass
        if validate_response(response.text):
            return response.text
            
        # Retry with simplified prompt
        retry_response = model.generate_content(
            f"""SIMPLIFIED ANALYSIS OF:
{truncated_prompt}

USE THIS TEMPLATE:
[SCALP] Direction|Confidence|Key Levels|ideal entry|stop|target
[SWING] Trend|Entry Zones|Risk|stop|target|confidence
[NOTES] Key Observations""",
            generation_config={"temperature": 0.2}
        )
        
        # Final validation check
        if validate_response(retry_response.text):
            return retry_response.text
            
        # Fallback to template
        return format_fallback_analysis("UNKNOWN")
            
    except google_exceptions.InvalidArgument as e:
        return f"⚠️ Analysis truncated due to length constraints"
    except Exception as e:
        return format_fallback_analysis("ERROR")
