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
    """More flexible validation with case-insensitive checks"""
    required_sections = [
        "SCALP", 
        "SWING", 
        "Quantitative Confidence",
        "AI Confidence",
        "PRICE ACTION NOTES",
        "CONFIDENCE GUARDRAILS"
    ]
    
    text_lower = text.lower()
    return all(section.lower() in text_lower for section in required_sections)

def get_gemini_analysis(prompt):
    try:
        truncated_prompt = textwrap.shorten(prompt, width=30000, placeholder="... [truncated]")
        
        response = model.generate_content(
            f"""STRICT RESPONSE FORMAT REQUIRED! ANALYZE BEARISH/BULLISH FACTORS:

{truncated_prompt}

*MANDATORY SECTIONS:*

🚀 SCALP (5-15m)
`Direction:` [LONG/SHORT/NONE]
`Confidence:` [%] (Technical/ML)
`Entry Zone:` LEVEL-LEVEL
`Targets:` LEVEL → LEVEL 
`Stop:` LEVEL

🌙 SWING (1-4H)  
`Direction:` [LONG/SHORT/NONE]
`Confidence:` [%] (Technical/ML)
`Key Levels:` Support: LEVEL | Resistance: LEVEL
`Entry Zone:` LEVEL-LEVEL
`Targets:` LEVEL-LEVEL
`Stop:` LEVEL
`Risk/Reward:` 1:X

🔍 MARKET DYNAMICS
1. Bullish Factors: [...] 
2. Bearish Risks: [...]

⚠️ ALERT: If conflicting signals, state: 'CONFLICTING SIGNALS - WAIT'"""
        )

        if validate_response(response.text):
            return response.text
        else:
            # Attempt recovery for common missing elements
            recovery_text = response.text.replace("Quant Confidence", "Quantitative Confidence")
            recovery_text = recovery_text.replace("AI Conf", "AI Confidence")
            if validate_response(recovery_text):
                return recovery_text
            return "⚠️ AI generated incomplete analysis. Check technicals manually."
            
    except google_exceptions.InvalidArgument as e:
        return f"❌ Message too long ({len(prompt)} chars). Max 30k characters."
    except Exception as e:
        return f"❌ API Error: {str(e)}"
