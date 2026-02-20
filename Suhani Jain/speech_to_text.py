import os
import logging
from dotenv import load_dotenv
from deepgram import DeepgramClient, FileSource, PrerecordedOptions

load_dotenv()
logger = logging.getLogger()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = DeepgramClient(DEEPGRAM_API_KEY)


async def transcribe_audio(file_path: str):
    logger.info("Sending audio to Deepgram STT")

    with open(file_path, "rb") as audio:
        buffer_data = audio.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
    )

    response =  deepgram.listen.prerecorded.v("1").transcribe_file(
        payload, options
    )

    transcript = response.results.channels[0].alternatives[0].transcript

    logger.info(f"Transcription: {transcript}")

    return transcript