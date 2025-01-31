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

def format_fallback_analysis(symbol):
    """Fallback template for failed AI analysis"""
    return f"""🔍 ANALYSIS FAILED - MANUAL CHECK REQUIRED

🚀 SCALP (5-15m)
`Quantitative Confidence:` [VALUE]%
`AI Confidence:` [VALUE]%
`Strength:` [▲ High/► Medium/▼ Low] (ADX VALUE)  
`Entry Strategy:` Manual Verification Needed
`Ideal Entry:` N/A → N/A  
`TP Levels:` N/A → N/A 
`Stop:` N/A  

🌙 SWING (1-4H)  
`Quantitative Confidence:` [VALUE]%
`AI Confidence:` [VALUE]%
`Market Structure:` [...] 
`Ideal Entry:` N/A → N/A  
`TP Levels:` N/A → N/A  
`SL:` N/A  

⚠️ CONFIDENCE GUARDRAILS  
- Verify volume and price action
- Check macroeconomic factors
- Confirm with on-chain data"""

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
[SCALP] Direction|Confidence|Key Levels
[SWING] Trend|Entry Zones|Risk
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
