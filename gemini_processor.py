# gemini_processor.py
import google.generativeai as genai
import textwrap
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=config.GEMINI_API_KEY)
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
            f"""ANALYSIS FORMAT REQUIREMENTS:
1. MUST INCLUDE ALL SECTIONS: SCALP, SWING, PRICE ACTION NOTES, CONFIDENCE GUARDRAILS
2. USE EXACT SECTION HEADERS FROM BELOW
3. ALWAYS SHOW QUANTITATIVE AND AI CONFIDENCE

{truncated_prompt}

*MANDATORY RESPONSE STRUCTURE:*

🚀 *SCALP (5-15m)*
`Quantitative Confidence:` [VALUE]%
`AI Confidence:` [VALUE]%
`Strength:` [▲ High/► Medium/▼ Low] (ADX VALUE)  
`Entry Strategy:`  
[🟢 Wait for retest of LEVEL | 🔴 Breakout above LEVEL | ⏳ Rebound from LEVEL]  
[🔴 Invalidation below LEVEL] 
`Ideal Entry:` LEVEL-LEVEL 
`Targets:` LEVEL → LEVEL  
`Stop:` LEVEL (-X.X%)  

🌙 *SWING (1-4H)*  
`Quantitative Confidence:` [VALUE]%
`AI Confidence:` [VALUE]%
`Market Structure: [...] 
[🔼 Breakout needed | ⏸️ Consolidation | 🔻 Pullback]  
`Ideal Entry:` LEVEL-LEVEL  
`TP Levels:` LEVEL → LEVEL  
`SL:` LEVEL (-X.X%)  

🔍 *PRICE ACTION NOTES*  
1. [...]
2. [...]

⚠️ *CONFIDENCE GUARDRAILS*  
- [...]"""
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
