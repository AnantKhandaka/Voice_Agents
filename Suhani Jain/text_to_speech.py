import os
import logging
from dotenv import load_dotenv
from deepgram import DeepgramClient, SpeakOptions

load_dotenv()
logger = logging.getLogger()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = DeepgramClient(DEEPGRAM_API_KEY)


async def speak_text(text: str, output_file="response.mp3"):
    logger.info("Converting text to speech via Deepgram")

    options = SpeakOptions(
        model="aura-asteria-en",
    )

    deepgram.speak.v("1").save(
        output_file,
        {"text": text},
        options
    )

    logger.info("Audio response saved")
    return output_file