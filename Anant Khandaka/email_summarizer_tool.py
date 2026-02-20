"""
Wrapper to call agent_email.py and return its output
This allows voice_assistant to use email summarization as a tool
"""

import asyncio
import sys
import io
import logging
from contextlib import redirect_stdout
from agent_email import fetch_latest_10_emails, summarize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_email_summaries_async() -> str:
    """
    Call agent_email.py logic and capture its output.
    Returns the exact same formatted summaries that agent_email.py prints.
    Fetches the latest 10 emails.
    """
    try:
        emails = fetch_latest_10_emails()
        
        if not emails:
            return "No emails found in inbox."
        
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            await summarize(emails)
        
        result = output_buffer.getvalue()
        
        if not result.strip():
            return "Error: No summary generated"
        
        return result
    
    except Exception as e:
        return f"Error getting email summaries: {str(e)}"


def get_email_summaries() -> str:
    """
    Get email summaries by calling agent_email.py logic.
    
    This function fetches the latest 10 emails and returns AI-generated summaries.
    It runs the exact same code as agent_email.py without any additional processing.
    
    Returns:
        Formatted email summaries from agent_email.py in the format:
        === SUMMARIES ===
        Email 1: ...
        Email 2: ...
        etc.
    """
    return asyncio.run(get_email_summaries_async())


if __name__ == "__main__":
    logger.info("Testing email summary wrapper...")
    result = get_email_summaries()
    logger.info(result)
