"""
Chunked Speech-to-Text using Deepgram Listen v2 (OPTIMIZED)

- Records 10-second audio chunks continuously
- Transcribes chunks in parallel with recording
- Detects actual silence (based on amplitude)
- Stops after 30 seconds of silence
- Returns only final transcripts per chunk
"""

import os
import time
import queue
import threading
import asyncio
import signal
import struct
import math
import io
import wave
import httpx
import pyaudio
from dotenv import load_dotenv


RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

CHUNK_SECONDS = 10
SILENCE_TIMEOUT = 30
SILENCE_THRESHOLD = 300  

load_dotenv()
DEEPGRAM_KEY = os.getenv("DEEPGRAM_KEY")


def pcm_to_wav(pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
    """Convert raw PCM data to WAV format with proper headers."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()



class AudioRecorder(threading.Thread):
    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.last_sound_time = time.time()

    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

        frames_per_chunk = int(RATE / FRAMES_PER_BUFFER * CHUNK_SECONDS)
        buffer = []

        try:
            while not self.stop_event.is_set():
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                buffer.append(data)
                
                
                samples = struct.unpack(f"{len(data)//2}h", data)  
                sum_squares = sum(s * s for s in samples)
                rms = math.sqrt(sum_squares / len(samples))
                
                if rms > SILENCE_THRESHOLD:
                    self.last_sound_time = time.time()

                
                if len(buffer) >= frames_per_chunk:
                    self.audio_queue.put(b"".join(buffer))
                    buffer.clear()

                if time.time() - self.last_sound_time >= SILENCE_TIMEOUT:
                    print("\n⏹️  30 seconds of silence detected, stopping...")
                    self.stop_event.set()
                    break

        finally:
            if buffer:
                self.audio_queue.put(b"".join(buffer))

            self.audio_queue.put(None)
            stream.stop_stream()
            stream.close()
            audio.terminate()



async def transcribe_chunk(audio_bytes: bytes, chunk_num: int) -> str:
    """Transcribe a single audio chunk using Deepgram REST API."""
    try:
        wav_data = pcm_to_wav(audio_bytes, RATE, CHANNELS)
        
        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": "nova-2",
            "smart_format": "true",
        }
        headers = {
            "Authorization": f"Token {DEEPGRAM_KEY}",
            "Content-Type": "audio/wav",
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, params=params, headers=headers, content=wav_data)
            response.raise_for_status()
            result = response.json()
        
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        
        if transcript:
            print(f"  ✅ Transcribed: '{transcript[:60]}...'" if len(transcript) > 60 else f"  ✅ Transcribed: '{transcript}'")
            return transcript
        else:
            print(f"  ⚠️  No speech in chunk {chunk_num}")
            return ""
            
    except httpx.HTTPStatusError as e:
        print(f"  ❌ HTTP error transcribing chunk {chunk_num}: {e.response.status_code}")
        return ""
    except Exception as e:
        print(f"  ❌ Error transcribing chunk {chunk_num}: {e}")
        return ""



def listen() -> str:
    """
    Listen to microphone and return complete transcription.
    Records in 10-second chunks while transcribing previous chunks in parallel.
    Stops after 30 seconds of silence.
    """
    stop_event = threading.Event()
    audio_queue: queue.Queue[bytes | None] = queue.Queue()
    full_transcript: list[str] = []

    def handle_sigint(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    recorder = AudioRecorder(audio_queue, stop_event)
    recorder.start()

    async def runner():
        """Process chunks as they arrive - runs in parallel with recording."""
        chunk_num = 0

        while not stop_event.is_set():
            
            chunk = audio_queue.get()
            if chunk is None:
                break

            chunk_num += 1
            print(f"🔄 Processing chunk {chunk_num}... (recording continues)")
            
            
            text = await transcribe_chunk(chunk, chunk_num)
            
            if text:
                print(f"✅ Chunk {chunk_num}: {text[:50]}..." if len(text) > 50 else f"✅ Chunk {chunk_num}: {text}")
                full_transcript.append(text)
            else:
                print(f"⚠️  Chunk {chunk_num}: No speech detected")

    asyncio.run(runner())
    recorder.join()

    return " ".join(full_transcript).strip()


async def listen_async() -> str:
    """
    Async version of listen() for use within existing event loops.
    Listen to microphone and return complete transcription.
    Records in 10-second chunks while transcribing previous chunks in parallel.
    Stops after 30 seconds of silence.
    """
    stop_event = threading.Event()
    audio_queue: queue.Queue[bytes | None] = queue.Queue()
    full_transcript: list[str] = []

    def handle_sigint(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    recorder = AudioRecorder(audio_queue, stop_event)
    recorder.start()

    chunk_num = 0

    while not stop_event.is_set():
        
        chunk = audio_queue.get()
        if chunk is None:
            break

        chunk_num += 1
        print(f"🔄 Processing chunk {chunk_num}... (recording continues)")
        
        
        text = await transcribe_chunk(chunk, chunk_num)
        
        if text:
            print(f"✅ Chunk {chunk_num}: {text[:50]}..." if len(text) > 50 else f"✅ Chunk {chunk_num}: {text}")
            full_transcript.append(text)
        else:
            print(f"⚠️  Chunk {chunk_num}: No speech detected")

    recorder.join()

    return " ".join(full_transcript).strip()


if __name__ == "__main__":
    print("🎤 Voice Recording Started")
    print("📦 Recording in 10-second chunks")
    print("⏸️  Auto-stops after 30 seconds of silence")
    print("🔄 Transcription happens in parallel with recording")
    print("─" * 50)

    result = listen()

    print("\n" + "─" * 50)
    if result:
        print("📝 FINAL TRANSCRIPTION:")
        print(result)
    else:
        print("⚠️  No speech detected")