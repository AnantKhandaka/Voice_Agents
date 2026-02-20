"""
Voice-Controlled AI Assistant
Uses Deepgram for voice input and Autogen agents with Ollama for processing
Now with TTS output and email summarization!
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from deepgram_STT import listen_async
from TTS_deepgram_simple import text_to_speech
from email_summarizer_tool import get_email_summaries
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
import subprocess
import platform

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def main():
    """Main voice assistant function."""
    logging.info("🎤 Voice Assistant Ready!")
    logging.info("=" * 60)
    logging.info("Speak your question or command...")
    logging.info("=" * 60)
    
    user_prompt = await listen_async()
    
    if not user_prompt:
        logging.error("No speech detected. Exiting.")
        return
    
    logging.info("=" * 60)
    logging.info(f"📝 You said: {user_prompt}")
    logging.info("=" * 60)
    
    model_client = OllamaChatCompletionClient(
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
    )
    
    email_tool = FunctionTool(
        get_email_summaries,
        description="Fetch and summarize the user's recent emails from their inbox. Only use when user explicitly asks about their emails or inbox."
    )
    
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[email_tool],
        system_message="""You are a helpful AI assistant. Answer all questions to the best of your knowledge.

You have access to a tool for checking emails:
- If the user asks about their emails/inbox, use the get_email_summaries tool and return its exact output
- For all other questions (movies, facts, general knowledge, etc.), answer directly using your knowledge

Be helpful and informative!""",
    )
    
    logging.info("🤖 AI Agent processing your request...")
    
    logging.info("=" * 60)
    logging.info("🤖 AI Response:")
    logging.info("=" * 60)
    
    result = await agent.run(task=user_prompt)
    
    full_response = ""
    if hasattr(result, 'messages') and result.messages:
        for msg in reversed(result.messages):
            if hasattr(msg, 'content') and msg.content and str(msg.content).strip():
                full_response = str(msg.content)
                break
    
    if not full_response:
        full_response = "I apologize, but I couldn't generate a response."
    
    print(full_response)
    logging.info("=" * 60)
    
    logging.info("🔊 Converting response to speech...")
    
    if len(full_response) > 1900:
        logging.warning(f"Response too long ({len(full_response)} chars), truncating for TTS...")
        full_response = full_response[:1900] + "..."
    
    audio_file = "response.wav"
    success = text_to_speech(full_response, audio_file)
    
    if success:
        logging.info(f"✅ Audio saved to {audio_file}")
        logging.info("🔊 Playing audio...")
        
        abs_path = os.path.abspath(audio_file)
        if not os.path.exists(abs_path):
            logging.error(f"Audio file not found: {abs_path}")
            return
        
        file_size = os.path.getsize(abs_path)
        logging.info(f"📊 Audio file size: {file_size} bytes")
        
        if file_size < 1000:
            logging.warning("Audio file seems too small, might be invalid")
        
        played = False
        
        if platform.system() == "Windows":
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(abs_path)
                pygame.mixer.music.play()
                
                import time
                duration = file_size / 32000
                logging.info(f"⏱️ Estimated duration: {duration:.1f} seconds")
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                pygame.mixer.quit()
                logging.info("✅ Audio played successfully (pygame)!")
                played = True
            except ImportError:
                logging.warning("pygame not installed, trying other methods...")
            except Exception as e1:
                logging.warning(f"pygame method failed: {e1}")
        
        if not played and platform.system() == "Windows":
            try:
                import winsound
                logging.info("🔊 Playing with winsound...")
                winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                logging.info("✅ Audio played successfully (winsound)!")
                played = True
            except Exception as e2:
                logging.warning(f"winsound method failed: {e2}")
        
        if not played and platform.system() == "Windows":
            try:
                logging.info("🔊 Playing with PowerShell...")
                cmd = f'''
                $player = New-Object System.Media.SoundPlayer("{abs_path}")
                $player.PlaySync()
                '''
                result = subprocess.run(
                    ["powershell", "-Command", cmd],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logging.info("✅ Audio played successfully (PowerShell)!")
                    played = True
                else:
                    logging.warning(f"PowerShell error: {result.stderr}")
            except Exception as e3:
                logging.warning(f"PowerShell method failed: {e3}")
        
        if not played:
            logging.warning("Could not auto-play audio with any method")
            logging.info(f"File location: {abs_path}")
            logging.info("You can play it manually with Windows Media Player")
            logging.info("To install pygame for better audio playback: pip install pygame")
        
        logging.info("=" * 60)
        logging.info("✅ Response complete!")
        logging.info("=" * 60)
    else:
        logging.error("Failed to convert text to speech")


if __name__ == "__main__":
    asyncio.run(main())
