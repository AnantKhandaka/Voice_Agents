"""
Simple Text-to-Speech using Deepgram REST API (synchronous)
No websockets, no async complexity - just send text, get audio
"""
import os
import logging
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

AUDIO_FILE = "output.wav"
TTS_TEXT = "Hello, this is a text to speech example using Deepgram. How are you doing today? I am fine thanks for asking."

def text_to_speech(text: str, output_file: str = AUDIO_FILE):
    """
    Convert text to speech using Deepgram REST API.
    
    Args:
        text: Text to convert to speech
        output_file: Where to save the audio file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        api_key = os.getenv("DEEPGRAM_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_KEY not set in .env file")
        
        logger.info("Starting TTS conversion...")
        
        url = "https://api.deepgram.com/v1/speak"
        
        params = {
            "model": "aura-asteria-en",  
            "encoding": "linear16",
            "sample_rate": 16000,
        }
        
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "text": text
        }
        
        logger.info(f"Sending request to Deepgram: '{text[:50]}...'")
        response = requests.post(url, params=params, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            logger.info(f"✅ Audio saved to {output_file} ({len(response.content)} bytes)")
            return True
        else:
            logger.error(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

def main():
    logger.info("="*60)
    logger.info("Text-to-Speech Conversion")
    logger.info("="*60)
    
    success = text_to_speech(TTS_TEXT, AUDIO_FILE)
    
    if success:
        print(f"\n✅ SUCCESS! Audio file created: {AUDIO_FILE}")
        print(f"📝 Text: {TTS_TEXT}")
    else:
        print("\n❌ FAILED to create audio file")

if __name__ == "__main__":
    main()
