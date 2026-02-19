import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.ollama import OllamaChatCompletionClient

try:
    from RAG_agent import (
        AdvancedDocumentParser,
        AdvancedChunker,
        OptimizedRAGIndexer,
        RAGAgentFactory
    )
except ImportError:
    from Agents.RAG_agent import (
        AdvancedDocumentParser,
        AdvancedChunker,
        OptimizedRAGIndexer,
        RAGAgentFactory
    )

app = FastAPI(
    title="RAG Agent API",
    description="Retrieval-Augmented Generation API for document analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentResponse(BaseModel):
    session_id: str
    filename: str
    doc_type: str
    chunks: int
    timestamp: str
    status: str

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class AnswerResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    documents: List[str]
    document_count: int
    total_chunks: int
    model: str
    created_at: str

class RefinedAnswer(BaseModel):
    """Refined answer with metadata"""
    question: str
    answer: str
    type: str  
    confidence: str  
    sources_found: int
    timestamp: str

class DocumentSummary(BaseModel):
    """Document summary response"""
    title: str
    summary: str
    key_topics: List[str]
    content_type: str
    pages: int

class SkillsResponse(BaseModel):
    """Skills extracted from document"""
    programming_languages: List[str]
    technical_skills: List[str]
    tools_and_libraries: List[str]
    certifications: List[str]
    specializations: List[str]

class ExperienceResponse(BaseModel):
    """Experience extracted from document"""
    title: str
    positions: List[str]
    duration: str
    key_achievements: List[str]

sessions = {}

@dataclass
class RAGSession:
    session_id: str
    agent: AssistantAgent
    memory: ChromaDBVectorMemory
    documents: List[str]
    total_chunks: int
    created_at: str
    model: str


@app.get("/")
async def serve_web():
    """Serve the web interface."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"error": "Web interface not found"}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "RAG Agent API",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/sessions/create")
async def create_session(model: str = "llama3.2:1b"):
    """Create a new RAG session."""
    session_id = str(uuid.uuid4())
    
    try:
        print(f"[DEBUG] Creating session {session_id} with model {model}")
        
        agent, memory = await RAGAgentFactory.create_agent(
            model=model,
            collection_name=f"rag_{session_id[:8]}"
        )
        
        print(f"[DEBUG] Agent and memory created successfully")
        
        await memory.clear()
        
        sessions[session_id] = RAGSession(
            session_id=session_id,
            agent=agent,
            memory=memory,
            documents=[],
            total_chunks=0,
            created_at=datetime.now().isoformat(),
            model=model
        )
        
        print(f"[DEBUG] Session {session_id} stored successfully")
        
        return {
            "session_id": session_id,
            "status": "created",
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Session creation failed: {error_details}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@app.post("/documents/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """Upload a document to a session."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    allowed_types = {".pdf", ".docx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Supported: {allowed_types}"
        )
    
    temp_dir = Path(tempfile.gettempdir()) / "rag_uploads" / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_path = temp_dir / file.filename
    
    try:
        content = await file.read()
        temp_path.write_bytes(content)
        
        indexer = OptimizedRAGIndexer(memory=session.memory, chunk_size=800)
        total_chunks, doc_count = await indexer.index_documents([str(temp_path)])
        
        if total_chunks == 0:
            temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Failed to extract content from document"
            )
        
        session.documents.append(file.filename)
        session.total_chunks += total_chunks
        
        return DocumentResponse(
            session_id=session_id,
            filename=file.filename,
            doc_type=file_ext[1:],
            chunks=total_chunks,
            timestamp=datetime.now().isoformat(),
            status="indexed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents."""
    
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    session = sessions[request.session_id]
    
    if not session.documents:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded to this session"
        )
    
    try:
        print(f"[DEBUG] Starting question: {request.question}")
        print(f"[DEBUG] Session documents: {session.documents}")
        print(f"[DEBUG] Session chunks: {session.total_chunks}")
        
        answer_text = ""
        
        stream = session.agent.run_stream(task=request.question)
        
        async for message in stream:
            if 'MemoryContent' in str(type(message)) or '[MemoryContent' in str(message):
                continue
            
            if hasattr(message, 'content'):
                content_str = str(message.content)
                
                if '[MemoryContent(' not in content_str:
                    answer_text += content_str
            elif isinstance(message, str) and '[MemoryContent(' not in message:
                answer_text += message
        
        print(f"[DEBUG] Got answer: {answer_text[:100]}...")
        
        answer_text = answer_text.strip()
        if "[MemoryContent(" in answer_text:
            import re
            
            pattern = r"content='([^']*(?:[^']|\'[^\'])*?)'\s*(?:,\s*mime_type|,\s*metadata|\)|$)"
            contents = re.findall(pattern, answer_text, re.DOTALL)
            
            if contents:
                
                answer_text = "\n".join([c.strip() for c in contents if c.strip()])
            else:
                
                contents = re.findall(r"MemoryContent\(content='([^']+)'", answer_text)
                if contents:
                    answer_text = "\n".join(contents)
        
        clean_answer = answer_text.strip()
        
        if "[MemoryContent(" in clean_answer:
            clean_answer = "[No valid answer - internal error]"
            print("[WARNING] MemoryContent slipped through to final answer!")
        
        useless_phrases = [
            "I'm not able to provide",
            "without more context",
            "I can suggest",
            "you need to",
            "try searching",
            "If you have",
            "the document is about",
            "[MemoryContent",
            "mime_type"
        ]
        
        for phrase in useless_phrases:
            if phrase.lower() in clean_answer.lower():
                better_prompt = f"""Based on the documents, answer this precisely: {request.question}
                
Provide a clear, direct answer without apologies or explanations. Just the facts. Do not include any memory objects or raw data."""
                answer_text = ""
                stream = session.agent.run_stream(task=better_prompt)
                async for message in stream:
                    if hasattr(message, 'content'):
                        answer_text += str(message.content)
                
                if "[MemoryContent(" in answer_text:
                    import re
                    contents = re.findall(r"content='([^']+)'|content=\"([^\"]+)\"", answer_text)
                    if contents:
                        answer_text = " ".join([c[0] if c[0] else c[1] for c in contents if c[0] or c[1]])
                
                clean_answer = answer_text.strip()
                break
        
        return RefinedAnswer(
            question=request.question,
            answer=clean_answer if clean_answer else "No relevant information found in documents",
            type="answer",
            confidence="high" if len(clean_answer) > 150 else "medium",
            sources_found=min(session.total_chunks, 5),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Question failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionInfo(
        session_id=session_id,
        documents=session.documents,
        document_count=len(session.documents),
        total_chunks=session.total_chunks,
        model=session.model,
        created_at=session.created_at
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "documents": s.documents,
                "document_count": len(s.documents),
                "model": s.model,
                "created_at": s.created_at
            }
            for sid, s in sessions.items()
        ]
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup resources."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions[session_id]
        
        await session.memory.close()
        
        temp_dir = Path(tempfile.gettempdir()) / "rag_uploads" / session_id
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        del sessions[session_id]
        
        return {
            "status": "deleted",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/batch")
async def ask_batch(session_id: str, questions: List[str]):
    """Ask multiple questions."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if not session.documents:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded to this session"
        )
    
    results = []
    
    for question in questions:
        try:
            print(f"[DEBUG] Processing question: {question}")
            answer_text = ""
            stream = session.agent.run_stream(task=question)
            
            async for message in stream:
                if 'MemoryContent' in str(type(message)) or '[MemoryContent' in str(message):
                    continue
                
                if hasattr(message, 'content'):
                    content_str = str(message.content)
                    if '[MemoryContent(' not in content_str:
                        answer_text += content_str
                elif isinstance(message, str) and '[MemoryContent(' not in message:
                    answer_text += message
            
            answer_text = answer_text.strip()
            if "[MemoryContent(" in answer_text:
                import re
                pattern = r"content='([^']*(?:[^']|\'[^\'])*?)'\s*(?:,\s*mime_type|,\s*metadata|\)|$)"
                contents = re.findall(pattern, answer_text, re.DOTALL)
                
                if contents:
                    answer_text = "\n".join([c.strip() for c in contents if c.strip()])
                else:
                    contents = re.findall(r"MemoryContent\(content='([^']+)'", answer_text)
                    if contents:
                        answer_text = "\n".join(contents)
            
            clean_answer = answer_text.strip() if answer_text else "No answer generated"
            

            if "[MemoryContent(" in clean_answer:
                clean_answer = "[No valid answer - internal error]"
                print(f"[WARNING] MemoryContent slipped through for question: {question}")
            
            results.append({
                "question": question,
                "answer": clean_answer,
                "status": "success"
            })
        except Exception as e:
            import traceback
            print(f"[ERROR] Question '{question}' failed: {str(e)}\n{traceback.format_exc()}")
            results.append({
                "question": question,
                "answer": None,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "session_id": session_id,
        "total_questions": len(questions),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup all sessions on shutdown."""
    for session_id in list(sessions.keys()):
        try:
            session = sessions[session_id]
            await session.memory.close()
            
            temp_dir = Path(tempfile.gettempdir()) / "rag_uploads" / session_id
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            
            del sessions[session_id]
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
