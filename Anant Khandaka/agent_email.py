import os
import imaplib
import email
import asyncio
import platform
import subprocess
import time
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken
from TTS_deepgram_simple import text_to_speech

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("⚠️ pygame not installed. Install with: pip install pygame")

load_dotenv()

EMAIL_ACCOUNT = os.getenv("JUMAIL")
APP_PASSWORD = os.getenv("JUMAIL_KEY")
IMAP_SERVER = "imap.gmail.com"

if not EMAIL_ACCOUNT or not APP_PASSWORD:
    raise ValueError("Email credentials not found. Check your .env file for JUMAIL and JUMAIL_KEY")

def fetch_latest_10_emails():
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, 993)
    mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    latest_10 = email_ids[-10:]

    emails_data = []
    for e_id in reversed(latest_10):
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        subject = msg["subject"]
        sender = msg["from"]

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        emails_data.append(f"From: {sender}\nSubject: {subject}\nBody: {body[:800]}")

    mail.logout()
    return emails_data

async def summarize(emails: list[str]) -> str:
    """
    Summarize emails and return the summary text.
    
    Returns:
        str: The summary text
    """
    logging.info("🤖 Generating AI summaries...")
    
    model_client = OllamaChatCompletionClient(
        model="qwen2.5:3b",
        base_url="http://localhost:11434"
    )

    assistant = AssistantAgent(
        name="email_ai",
        model_client=model_client,
        system_message="You are a helpful assistant that summarizes emails concisely."
    )

    prompt = "\n\n".join(
        f"Email {i+1}:\n{mail}" for i, mail in enumerate(emails)
    )

    task_msg = TextMessage(content=(
        "Summarize each of the following emails in 1–2 concise lines:\n\n" + prompt
    ), source="user")

    cancellation_token = CancellationToken()
    response = await assistant.on_messages([task_msg], cancellation_token)

    summary_text = ""
    if hasattr(response, 'inner_messages') and response.inner_messages:
        for msg in reversed(response.inner_messages):
            if hasattr(msg, 'content') and msg.content and msg.content != "TERMINATE":
                summary_text = msg.content
                break
    else:
        summary_text = response.chat_message.content  # 
    print(" EMAIL SUMMARIES ")
    print("\n")
    print(summary_text)
    
    return summary_text

async def main():
    logging.info("📧 Fetching latest 10 emails...")
    emails = fetch_latest_10_emails()
    
    logging.info(f"✅ Retrieved {len(emails)} emails")
    
    
    summary_text = await summarize(emails)
    
    logging.info("🔊 Converting summaries to speech...")
    
    if len(summary_text) > 1900:
        summary_text = summary_text[:1900] + "..."
    
    audio_file = "email_summaries.wav"
    success = text_to_speech(summary_text, audio_file)
    
    if not success:
        logging.error("❌ Failed to generate speech.")
        return
    
    abs_path = os.path.abspath(audio_file)
    logging.info(f"✅ Audio saved: {abs_path}")
    
    logging.info("🎵 Playing audio...")
    
    if PYGAME_AVAILABLE:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(abs_path)
            pygame.mixer.music.play()
            
            logging.info("▶️ Audio is playing...")
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.quit()
            logging.info("✅ Audio playback completed!")
            
        except Exception as e:
            logging.error(f"❌ pygame error: {e}")
            logging.warning("⚠️ Trying alternative playback method...")
            
            if platform.system() == "Windows":
                try:
                    import winsound
                    winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                    logging.info("✅ Audio played using winsound")
                except Exception as e2:
                    logging.error(f"❌ winsound error: {e2}")
                    logging.warning(f"⚠️ Could not auto-play audio. Please play manually: {abs_path}")
    else:
        logging.warning("⚠️ pygame not available. Attempting alternative playback...")
        
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                logging.info("✅ Audio played using winsound")
            except Exception as e:
                logging.error(f"❌ Error: {e}")
                logging.warning(f"⚠️ Could not auto-play audio. Please play manually: {abs_path}")
    
    logging.info("=" * 60)
    logging.info("✅ Email summaries complete!")
    logging.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
