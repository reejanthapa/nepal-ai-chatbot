# fastapi_backend.py
"""
FastAPI Backend for Nepal AI Chatbot
API endpoints for native Android integration (no Streamlit dependency)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import time
import uuid
from contextlib import asynccontextmanager

# Import your existing chatbot components
from main import NepalRAGChatbot  # Using your main.py file
from language_manager import Language, LanguageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
nepal_chatbot = None

# ✅ UPDATED: Request/Response Models to match your Android data classes exactly
class ChatRequest(BaseModel):
    message: str
    languagePreference: Optional[str] = None  # ✅ Direct camelCase field
    sessionId: Optional[str] = None          # ✅ Direct camelCase field

class ChatResponse(BaseModel):
    response: str
    confidence: float
    responseTime: float      # ✅ Direct camelCase field
    languageUsed: str       # ✅ Direct camelCase field
    sourcesUsed: int        # ✅ Direct camelCase field
    promptStrategy: str     # ✅ Direct camelCase field
    sessionId: str          # ✅ Direct camelCase field
    isLanguageSwitch: Optional[bool] = False  # ✅ Direct camelCase field

class LanguageSelectionRequest(BaseModel):
    languageChoice: str     # ✅ Direct camelCase field
    sessionId: Optional[str] = None  # ✅ Direct camelCase field

class LanguageSelectionResponse(BaseModel):
    success: bool
    selectedLanguage: str    # ✅ Direct camelCase field
    confirmationMessage: str # ✅ Direct camelCase field
    sessionId: str          # ✅ Direct camelCase field

class SystemStatus(BaseModel):
    status: str
    systemInitialized: bool  # ✅ Direct camelCase field
    datasetSize: int        # ✅ Direct camelCase field
    model: str
    embeddingMethod: str    # ✅ Direct camelCase field
    totalQueries: int       # ✅ Direct camelCase field
    successRate: float      # ✅ Direct camelCase field

class ExampleQuestionsResponse(BaseModel):
    questions: List[str]
    language: str

# Session management (simple in-memory store)
user_sessions = {}

def clean_old_sessions():
    """Clean sessions older than 24 hours"""
    current_time = time.time()
    sessions_to_remove = []
   
    for session_id, session_data in user_sessions.items():
        if current_time - session_data['created_at'] > 86400:  # 24 hours
            sessions_to_remove.append(session_id)
   
    for session_id in sessions_to_remove:
        del user_sessions[session_id]

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global nepal_chatbot
    try:
        logger.info("Initializing Nepal RAG Chatbot system...")
        nepal_chatbot = NepalRAGChatbot()
        logger.info("System initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise
   
    yield
   
    # Shutdown
    logger.info("Shutting down Nepal RAG Chatbot system...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Nepal AI Chatbot API",
    description="Advanced RAG-powered chatbot API for Nepal with bilingual support",
    version="1.0.0",
    lifespan=lifespan
)

# Enhanced CORS middleware for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://10.0.2.2:*",
        "http://192.168.*.*",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add a simple OPTIONS handler for preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}

@app.get("/", response_model=Dict[str, str])
async def health_check():
    return {
        "message": "🇳🇵 Nepal AI Chatbot API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/test", response_model=Dict[str, str])
async def test_endpoint():
    return {
        "status": "success",
        "message": "Connection to Nepal Chatbot API successful!",
        "timestamp": str(time.time()),
        "server": "FastAPI"
    }

@app.get("/health", response_model=SystemStatus)
async def get_system_status():
    if not nepal_chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
   
    metrics = nepal_chatbot.get_system_metrics()
   
    return SystemStatus(
        status="healthy",
        systemInitialized=True,
        datasetSize=metrics.get('dataset_size', 0),
        model=metrics.get('model', 'gemini-1.5-flash'),
        embeddingMethod=metrics.get('embedding_method', 'hybrid'),
        totalQueries=metrics.get('total_queries', 0),
        successRate=metrics.get('success_rate', 0.0)
    )

@app.post("/language/select", response_model=LanguageSelectionResponse)
async def select_language(request: LanguageSelectionRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(clean_old_sessions)
   
    session_id = request.sessionId or str(uuid.uuid4())
   
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'language_manager': LanguageManager(),
            'message_history': [],
            'created_at': time.time()
        }
   
    session = user_sessions[session_id]
   
    try:
        selected_language = session['language_manager'].parse_language_selection(request.languageChoice)
        session['language_manager'].set_user_preference(selected_language)
       
        confirmation = session['language_manager'].get_language_switch_confirmation(selected_language)
       
        return LanguageSelectionResponse(
            success=True,
            selectedLanguage=selected_language.value,
            confirmationMessage=confirmation['message'],
            sessionId=session_id
        )
    except Exception as e:
        logger.error(f"Language selection failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Language selection failed: {str(e)}"
        )

@app.get("/language/selection-message", response_model=Dict[str, str])
async def get_language_selection_message():
    lang_manager = LanguageManager()
    messages = lang_manager.get_language_selection_message()
    return {
        "english": messages['english'],
        "nepali": messages['nepali']
    }

@app.get("/examples", response_model=ExampleQuestionsResponse)
async def get_example_questions(language: str = "mixed"):
    try:
        from config import EXAMPLE_QUESTIONS
       
        if language.lower() == "nepali":
            nepali_questions = [q for q in EXAMPLE_QUESTIONS if any(char in q for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')]
            return ExampleQuestionsResponse(questions=nepali_questions, language="nepali")
        elif language.lower() == "english":
            english_questions = [q for q in EXAMPLE_QUESTIONS if not any(char in q for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')]
            return ExampleQuestionsResponse(questions=english_questions, language="english")
        else:
            return ExampleQuestionsResponse(questions=EXAMPLE_QUESTIONS, language="mixed")
    except ImportError:
        fallback_questions = [
            "What are the best places to visit in Nepal?",
            "Tell me about Nepali culture and festivals",
            "How do I get a visa for Nepal?",
            "What is the weather like in Nepal?",
            "नेपालमा घुम्न कुन ठाउँ राम्रो छ?",
            "नेपाली संस्कृति र चाडपर्वको बारेमा बताउनुहोस्",
            "नेपालमा मौसम कस्तो छ?"
        ]
        return ExampleQuestionsResponse(questions=fallback_questions, language="mixed")

# ✅ COMPLETELY FIXED: Chat endpoint with proper field mapping
@app.post("/chat", response_model=ChatResponse)
async def send_message(request: ChatRequest, background_tasks: BackgroundTasks):
    if not nepal_chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
   
    background_tasks.add_task(clean_old_sessions)
   
    try:
        session_id = request.sessionId or str(uuid.uuid4())
       
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                'language_manager': LanguageManager(),
                'message_history': [],
                'created_at': time.time()
            }
       
        session = user_sessions[session_id]
       
        if request.languagePreference:
            if request.languagePreference.lower() == 'nepali':
                session['language_manager'].set_user_preference(Language.NEPALI)
            elif request.languagePreference.lower() == 'english':
                session['language_manager'].set_user_preference(Language.ENGLISH)
       
        original_lang_manager = nepal_chatbot.language_manager
        nepal_chatbot.language_manager = session['language_manager']
       
        result = nepal_chatbot.process_query(request.message)
        nepal_chatbot.language_manager = original_lang_manager
       
        session['message_history'].append({
            'user_message': request.message,
            'bot_response': result['response'],
            'timestamp': time.time(),
            'language': result.get('response_language', 'auto')
        })
       
        if len(session['message_history']) > 20:
            session['message_history'] = session['message_history'][-20:]
       
        # ✅ FIXED: Use direct camelCase field names that match your Android models
        return ChatResponse(
            response=result['response'],
            confidence=result['confidence'],
            responseTime=result['response_time'],                    # ✅ snake_case -> camelCase
            languageUsed=result.get('response_language', 'auto'),   # ✅ snake_case -> camelCase
            sourcesUsed=result['sources_used'],                     # ✅ snake_case -> camelCase
            promptStrategy=result['prompt_strategy'],               # ✅ snake_case -> camelCase
            sessionId=session_id,                                   # ✅ snake_case -> camelCase
            isLanguageSwitch=result.get('language_switch', False)   # ✅ snake_case -> camelCase
        )
       
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/session/{session_id}/history", response_model=List[Dict[str, Any]])
async def get_chat_history(session_id: str):
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
   
    return user_sessions[session_id]['message_history']

@app.delete("/session/{sessionId}")
async def clear_session(sessionId: str):
    if sessionId in user_sessions:
        del user_sessions[sessionId]
        return {"success": True, "message": "Session cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics():
    if not nepal_chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
   
    metrics = nepal_chatbot.get_system_metrics()
    metrics['active_sessions'] = len(user_sessions)
    metrics['total_messages'] = sum(len(session['message_history']) for session in user_sessions.values())
    return metrics

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

@app.get("/debug/routes")
async def debug_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting Nepal AI Chatbot API Server...")
    print("=" * 60)
    print("📱 Android Emulator: http://10.0.2.2:8000")
    print("💻 Localhost: http://localhost:8000")
    print("📋 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🧪 Test Endpoint: http://localhost:8000/test")
    print("🛠️  Debug Routes: http://localhost:8000/debug/routes")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )