# AI Agent Projects

This folder contains multiple AI agent applications built with AutoGen, Deepgram, and Ollama.

## 📦 Projects Included

### 1. **Voice Assistant** (`voice_assistant.py`)
- Voice-controlled AI assistant using Deepgram for speech-to-text
- Integration with Ollama LLM (qwen2.5:3b)
- Text-to-speech output
- Email summarization capability

### 2. **Email Agent** (`agent_email.py`)
- Fetches latest emails from Gmail
- AI-powered email summarization
- Audio playback of summaries

### 3. **RAG Agent** (`RAG_agent.py`)
- Retrieval-Augmented Generation system
- Supports multiple document types (PDF, DOCX, TXT, URLs)
- ChromaDB vector database integration
- Advanced document parsing and chunking

### 4. **RAG API** (`rag_api.py`)
- FastAPI web service for RAG functionality
- REST API endpoints for document upload and querying
- Web interface included (`index.html`)

### 5. **Speech-to-Text** (`deepgram_STT.py`)
- Chunked audio recording and transcription
- Silence detection
- Parallel processing for efficiency

### 6. **Text-to-Speech** (`TTS_deepgram_simple.py`)
- Simple TTS using Deepgram API
- WAV audio output

## 🛠️ Setup Instructions

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running locally
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull the required model
   ollama pull qwen2.5:3b
   ```

3. **API Keys Required:**
   - Deepgram API key (for voice features)
   - Gmail App Password (for email features)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Github/Agents
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_voice_assistant.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the `Agents` folder:
   ```env
   DEEPGRAM_KEY=your_deepgram_api_key_here
   JUMAIL=your_email@gmail.com
   JUMAIL_KEY=your_gmail_app_password_here
   ```

   **How to get Gmail App Password:**
   - Go to Google Account settings
   - Enable 2-Factor Authentication
   - Generate an App Password for "Mail"

## 🚀 Usage

### Voice Assistant
```bash
python voice_assistant.py
```
Speak naturally and ask questions or request email summaries.

### Email Agent (Standalone)
```bash
python agent_email.py
```
Fetches and summarizes your latest 10 emails.

### RAG Agent (Interactive)
```bash
python RAG_agent.py
```
Input documents and ask questions about their content.

### RAG API Server
```bash
python rag_api.py
```
Then open `http://localhost:8000` in your browser.

## 📋 Dependencies

See `requirements_voice_assistant.txt` for full list. Key dependencies:
- `autogen-agentchat`, `autogen-core`, `autogen-ext`
- `deepgram-sdk`
- `fastapi`, `uvicorn`
- `pyaudio`, `pygame`
- `pdfplumber`, `python-docx`

## 🔧 Troubleshooting

### PyAudio Installation Issues (Windows)
```bash
pip install pipwin
pipwin install pyaudio
```

### Ollama Connection Issues
Ensure Ollama is running:
```bash
ollama serve
```

### Audio Device Issues
Make sure your microphone is properly connected and permissions are granted.

## 📝 Notes

- All projects use Ollama for local LLM inference (no OpenAI API needed)
- Voice features require Deepgram API (has free tier)
- Email features work with Gmail (requires App Password)
- RAG features store data in local ChromaDB

## 🤝 Contributing

Feel free to improve any of these agents or add new features!

## 📄 License

[Specify your license here]

---
*Developed as part of AI Agents Internship Program - February 2026*
