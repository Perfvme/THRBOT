# api_test.py
import requests
import config

print("🟠 Script started")  # Debug line 1

def test_deepseek_api():
    print("🟠 Entering API test function")  # Debug line 2
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"
    }
    
    test_payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Test"}]
    }
    
    try:
        print("🟠 Attempting API call...")  # Debug line 3
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=test_payload,
            headers=headers,
            timeout=10
        )
        print("🟠 Received API response")  # Debug line 4
        
    except Exception as e:
        print(f"🔴 Exception occurred: {str(e)}")  # Enhanced error reporting
        return

    print("🟢 Final status code:", response.status_code)  # Debug line 5

if __name__ == "__main__":
    print("🟠 Main block executed")  # Debug line 6
    test_deepseek_api()
    print("🟠 Script completed")  # Debug line 7