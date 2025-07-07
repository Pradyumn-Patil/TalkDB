import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api_key():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("ERROR: GEMINI_API_KEY not found in .env file")
            return False
            
        # Configure API
        genai.configure(api_key=api_key)
        
        # Try to create model
        model = genai.GenerativeModel('gemini-pro')
        
        # Test simple generation
        response = model.generate_content('Test')
        
        print("SUCCESS: API key is valid and working")
        print(f"Test response: {response.text}")
        return True
        
    except Exception as e:
        print(f"ERROR: API test failed - {str(e)}")
        return False

if __name__ == "__main__":
    test_api_key()
