import asyncio
import sounddevice as sd
from scipy.io.wavfile import write
import os
from dotenv import load_dotenv

from rag_tool.rag_engine import RAGEngine
from rag_tool.logger_config import setup_logger
from voice.speech_to_text import transcribe_audio
from voice.text_to_speech import speak_text

load_dotenv()
logger = setup_logger()

rag_engine = RAGEngine()

def record_audio(filename="input.wav", duration=5, fs=44100):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("✅ Recording complete")
    return filename


async def main():
    print("🎙 Voice RAG Assistant Ready")

    while True:
        input("Press Enter to ask a question (or Ctrl+C to exit)...")

        audio_file = record_audio()

        # 1️⃣ Speech to Text
        question = await transcribe_audio(audio_file)
        print("You asked:", question)

        # 2️⃣ RAG Answer
        answer = rag_engine.query(question)
        print("Answer:", answer)

        # 3️⃣ Text to Speech
        output_audio = await speak_text(answer)

        os.system(f"start {output_audio}")  # Windows playback


if __name__ == "__main__":
    asyncio.run(main())