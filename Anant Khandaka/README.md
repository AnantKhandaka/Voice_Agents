# AI Agent Projects

Multiple AI agent applications built with AutoGen, Deepgram, and Ollama.

## Projects

1. Voice Assistant - voice_assistant.py
2. Email Agent - agent_email.py
3. RAG Agent - RAG_agent.py
4. RAG API - rag_api.py
5. Speech-to-Text - deepgram_STT.py
6. Text-to-Speech - TTS_deepgram_simple.py

## Setup

1. Install Python 3.8+
2. Install Ollama and pull model:
   ```
   ollama pull qwen2.5:3b
   ```
3. Install dependencies:
   ```
   pip install -r requirements_voice_assistant.txt
   ```
4. Create .env file with:
   ```
   DEEPGRAM_KEY=your_key
   JUMAIL=your_email@gmail.com
   JUMAIL_KEY=your_app_password
   ```

## Usage

Voice Assistant:
```
python voice_assistant.py
```

Email Agent:
```
python agent_email.py
```

RAG Agent:
```
python RAG_agent.py
```

RAG API:
```
python rag_api.py
```

## Dependencies

See requirements_voice_assistant.txt
