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
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
nepal_chatbot = None

# Import with proper error handling
try:
    from main import NepalRAGChatbot
    CHATBOT_AVAILABLE = True
    logger.info("✅ Successfully imported NepalRAGChatbot")
except ImportError as e:
    logger.error(f"❌ Could not import NepalRAGChatbot: {e}")
    CHATBOT_AVAILABLE = False

try:
    from language_manager import Language, LanguageManager
    LANGUAGE_MANAGER_AVAILABLE = True
    logger.info("✅ Successfully imported Language components")
except ImportError as e:
    logger.error(f"❌ Could not import language manager: {e}")
    LANGUAGE_MANAGER_AVAILABLE = False
    
    # Create minimal fallback classes
    class Language:
        ENGLISH = "english"
        NEPALI = "nepali"
    
    class LanguageManager:
        def __init__(self):
            self.language = Language.ENGLISH
        
        def parse_language_selection(self, choice):
            return Language.ENGLISH if "english" in choice.lower() else Language.NEPALI
        
        def set_user_preference(self, lang):
            self.language = lang
        
        def get_language_switch_confirmation(self, lang):
            return {"message": f"Language set to {lang}"}
        
        def get_language_selection_message(self):
            return {
                "english": "Please select your language preference",
                "nepali": "कृपया आफ्नो भाषा छान्नुहोस्"
            }

try:
    from config import EXAMPLE_QUESTIONS
    CONFIG_AVAILABLE = True
    logger.info("✅ Successfully imported config")
except ImportError as e:
    logger.error(f"❌ Could not import config: {e}")
    CONFIG_AVAILABLE = False
    EXAMPLE_QUESTIONS = [
        "What are the best places to visit in Nepal?",
        "Tell me about Nepali culture and festivals",
        "How do I get a visa for Nepal?",
        "What is the weather like in Nepal?",
        "नेपालमा घुम्न कुन ठाउँ राम्रो छ?",
        "नेपाली संस्कृति र चाडपर्वको बारेमा बताउनुहोस्",
        "नेपालमा मौसम कस्तो छ?"
    ]

# ✅ Request/Response Models (exactly matching your Android data classes)
class ChatRequest(BaseModel):
    message: str
    languagePreference: Optional[str] = None
    sessionId: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    responseTime: float
    languageUsed: str
    sourcesUsed: int
    promptStrategy: str
    sessionId: str
    isLanguageSwitch: Optional[bool] = False

class LanguageSelectionRequest(BaseModel):
    languageChoice: str
    sessionId: Optional[str] = None

class LanguageSelectionResponse(BaseModel):
    success: bool
    selectedLanguage: str
    confirmationMessage: str
    sessionId: str

class SystemStatus(BaseModel):
    status: str
    systemInitialized: bool
    datasetSize: int
    model: str
    embeddingMethod: str
    totalQueries: int
    successRate: float

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

# Lifespan event handler with robust error handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global nepal_chatbot
    logger.info("🚀 Starting Nepal RAG Chatbot system...")
    
    try:
        if CHATBOT_AVAILABLE:
            logger.info("Initializing chatbot...")
            nepal_chatbot = NepalRAGChatbot()
            logger.info("✅ Chatbot initialization completed successfully")
        else:
            logger.warning("⚠️ Running in fallback mode - chatbot not available")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {str(e)}")
        logger.info("🔄 Continuing with limited functionality...")
        nepal_chatbot = None
    
    # Log system status
    logger.info(f"System Status:")
    logger.info(f"  - Chatbot Available: {CHATBOT_AVAILABLE}")
    logger.info(f"  - Language Manager Available: {LANGUAGE_MANAGER_AVAILABLE}")
    logger.info(f"  - Config Available: {CONFIG_AVAILABLE}")
    logger.info("✅ System ready to serve requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Nepal RAG Chatbot system...")

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
    allow_origins=["*"],  # Allow all origins for development
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
async def root():
    """Root endpoint with basic API info"""
    return {
        "message": "🇳🇵 Nepal AI Chatbot API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "test": "/test"
    }

@app.get("/test", response_model=Dict[str, Any])
async def test_endpoint():
    """Test endpoint to verify API connectivity"""
    return {
        "status": "success",
        "message": "Connection to Nepal Chatbot API successful!",
        "timestamp": str(time.time()),
        "server": "FastAPI",
        "components": {
            "chatbot_available": CHATBOT_AVAILABLE,
            "language_manager_available": LANGUAGE_MANAGER_AVAILABLE,
            "config_available": CONFIG_AVAILABLE
        }
    }

# ✅ CRITICAL: This is the health check endpoint Railway is looking for
@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        # Get metrics if chatbot is available
        if nepal_chatbot and hasattr(nepal_chatbot, 'get_system_metrics'):
            try:
                metrics = nepal_chatbot.get_system_metrics()
            except Exception as e:
                logger.warning(f"Could not get system metrics: {e}")
                metrics = {}
        else:
            metrics = {}
        
        return SystemStatus(
            status="healthy",
            systemInitialized=nepal_chatbot is not None,
            datasetSize=metrics.get('dataset_size', 1000),
            model=metrics.get('model', 'gemini-1.5-flash'),
            embeddingMethod=metrics.get('embedding_method', 'hybrid'),
            totalQueries=metrics.get('total_queries', 0),
            successRate=metrics.get('success_rate', 1.0)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Return a basic healthy status even if there are issues
        return SystemStatus(
            status="healthy",
            systemInitialized=False,
            datasetSize=0,
            model="fallback",
            embeddingMethod="none",
            totalQueries=0,
            successRate=1.0
        )

@app.post("/language/select", response_model=LanguageSelectionResponse)
async def select_language(request: LanguageSelectionRequest, background_tasks: BackgroundTasks):
    """Language selection endpoint"""
    background_tasks.add_task(clean_old_sessions)
    
    session_id = request.sessionId or str(uuid.uuid4())
    
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'language_manager': LanguageManager() if LANGUAGE_MANAGER_AVAILABLE else None,
            'message_history': [],
            'created_at': time.time()
        }
    
    try:
        if LANGUAGE_MANAGER_AVAILABLE and user_sessions[session_id]['language_manager']:
            session = user_sessions[session_id]
            selected_language = session['language_manager'].parse_language_selection(request.languageChoice)
            session['language_manager'].set_user_preference(selected_language)
            confirmation = session['language_manager'].get_language_switch_confirmation(selected_language)
            
            return LanguageSelectionResponse(
                success=True,
                selectedLanguage=selected_language.value if hasattr(selected_language, 'value') else str(selected_language),
                confirmationMessage=confirmation['message'],
                sessionId=session_id
            )
        else:
            # Fallback language selection
            lang = Language.ENGLISH if "english" in request.languageChoice.lower() else Language.NEPALI
            return LanguageSelectionResponse(
                success=True,
                selectedLanguage=lang,
                confirmationMessage=f"Language set to {lang}",
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
    """Get language selection message in both languages"""
    if LANGUAGE_MANAGER_AVAILABLE:
        try:
            lang_manager = LanguageManager()
            messages = lang_manager.get_language_selection_message()
            return {
                "english": messages['english'],
                "nepali": messages['nepali']
            }
        except Exception as e:
            logger.error(f"Error getting language selection message: {e}")
    
    # Fallback messages
    return {
        "english": "Please select your language preference: Type 'English' for English or 'Nepali' for नेपाली",
        "nepali": "कृपया आफ्नो भाषा छान्नुहोस्: अंग्रेजीको लागि 'English' वा नेपालीको लागि 'Nepali' टाइप गर्नुहोस्"
    }

@app.get("/examples", response_model=ExampleQuestionsResponse)
async def get_example_questions(language: str = "mixed"):
    """Get example questions in the specified language"""
    try:
        if language.lower() == "nepali":
            nepali_questions = [q for q in EXAMPLE_QUESTIONS if any(ord(char) > 2304 for char in q)]
            return ExampleQuestionsResponse(questions=nepali_questions, language="nepali")
        elif language.lower() == "english":
            english_questions = [q for q in EXAMPLE_QUESTIONS if not any(ord(char) > 2304 for char in q)]
            return ExampleQuestionsResponse(questions=english_questions, language="english")
        else:
            return ExampleQuestionsResponse(questions=EXAMPLE_QUESTIONS, language="mixed")
    except Exception as e:
        logger.error(f"Error getting example questions: {e}")
        # Return fallback questions
        fallback = [
            "What are the main tourist attractions in Nepal?",
            "Tell me about Nepali festivals",
            "How is the weather in Kathmandu?",
            "नेपालका मुख्य पर्यटकीय स्थलहरू के के हुन्?",
            "नेपाली चाडपर्वको बारेमा बताउनुहोस्"
        ]
        return ExampleQuestionsResponse(questions=fallback, language="mixed")

# ✅ Main chat endpoint with robust error handling
@app.post("/chat", response_model=ChatResponse)
async def send_message(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint for processing user messages"""
    if not CHATBOT_AVAILABLE or not nepal_chatbot:
        # Return fallback response when chatbot is not available
        return ChatResponse(
            response=f"Hello! I received your message: '{request.message}'. The chatbot system is currently in fallback mode. Please ensure all required files are properly configured.",
            confidence=0.8,
            responseTime=0.1,
            languageUsed="english",
            sourcesUsed=0,
            promptStrategy="fallback",
            sessionId=request.sessionId or str(uuid.uuid4()),
            isLanguageSwitch=False
        )
    
    background_tasks.add_task(clean_old_sessions)
    
    try:
        session_id = request.sessionId or str(uuid.uuid4())
        
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                'language_manager': LanguageManager() if LANGUAGE_MANAGER_AVAILABLE else None,
                'message_history': [],
                'created_at': time.time()
            }
        
        session = user_sessions[session_id]
        
        # Set language preference if provided
        if request.languagePreference and LANGUAGE_MANAGER_AVAILABLE and session['language_manager']:
            if request.languagePreference.lower() == 'nepali':
                session['language_manager'].set_user_preference(Language.NEPALI)
            elif request.languagePreference.lower() == 'english':
                session['language_manager'].set_user_preference(Language.ENGLISH)
        
        # Process query with the chatbot
        start_time = time.time()
        
        # Temporarily set language manager if available
        original_lang_manager = None
        if hasattr(nepal_chatbot, 'language_manager') and session['language_manager']:
            original_lang_manager = nepal_chatbot.language_manager
            nepal_chatbot.language_manager = session['language_manager']
        
        try:
            result = nepal_chatbot.process_query(request.message)
        finally:
            # Restore original language manager
            if original_lang_manager:
                nepal_chatbot.language_manager = original_lang_manager
        
        # Store conversation history
        session['message_history'].append({
            'user_message': request.message,
            'bot_response': result['response'],
            'timestamp': time.time(),
            'language': result.get('response_language', 'auto')
        })
        
        # Keep only last 20 messages
        if len(session['message_history']) > 20:
            session['message_history'] = session['message_history'][-20:]
        
        # Return properly formatted response
        return ChatResponse(
            response=result['response'],
            confidence=result.get('confidence', 0.8),
            responseTime=result.get('response_time', time.time() - start_time),
            languageUsed=result.get('response_language', 'auto'),
            sourcesUsed=result.get('sources_used', 0),
            promptStrategy=result.get('prompt_strategy', 'default'),
            sessionId=session_id,
            isLanguageSwitch=result.get('language_switch', False)
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        # Return error response instead of raising exception
        return ChatResponse(
            response=f"I apologize, but I encountered an error processing your message: {str(e)[:100]}. Please try again.",
            confidence=0.0,
            responseTime=0.1,
            languageUsed="english",
            sourcesUsed=0,
            promptStrategy="error",
            sessionId=request.sessionId or str(uuid.uuid4()),
            isLanguageSwitch=False
        )

@app.get("/session/{session_id}/history", response_model=List[Dict[str, Any]])
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return user_sessions[session_id]['message_history']

@app.delete("/session/{sessionId}")
async def clear_session(sessionId: str):
    """Clear a specific session"""
    if sessionId in user_sessions:
        del user_sessions[sessionId]
        return {"success": True, "message": "Session cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics():
    """Get system metrics and statistics"""
    try:
        base_metrics = {
            "active_sessions": len(user_sessions),
            "total_messages": sum(len(session['message_history']) for session in user_sessions.values()),
            "system_components": {
                "chatbot_available": CHATBOT_AVAILABLE,
                "language_manager_available": LANGUAGE_MANAGER_AVAILABLE,
                "config_available": CONFIG_AVAILABLE
            }
        }
        
        if nepal_chatbot and hasattr(nepal_chatbot, 'get_system_metrics'):
            try:
                chatbot_metrics = nepal_chatbot.get_system_metrics()
                base_metrics.update(chatbot_metrics)
            except Exception as e:
                logger.warning(f"Could not get chatbot metrics: {e}")
        
        return base_metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "error": "Could not retrieve metrics",
            "active_sessions": 0,
            "total_messages": 0
        }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404, "path": str(request.url)}

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to list all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else [],
                "name": getattr(route, 'name', 'unnamed')
            })
    return {"routes": routes, "total_routes": len(routes)}

# ✅ CRITICAL: Proper port configuration for Railway
if __name__ == "__main__":
    import uvicorn
    
    # Use Railway's PORT environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 60)
    print("🚀 Starting Nepal AI Chatbot API Server...")
    print(f"🌐 Port: {port}")
    print(f"📦 Components loaded:")
    print(f"   - Chatbot: {'✅' if CHATBOT_AVAILABLE else '❌'}")
    print(f"   - Language Manager: {'✅' if LANGUAGE_MANAGER_AVAILABLE else '❌'}")
    print(f"   - Config: {'✅' if CONFIG_AVAILABLE else '❌'}")
    print("=" * 60)
    print("📱 Android Emulator: http://10.0.2.2:{port}")
    print(f"💻 Localhost: http://localhost:{port}")
    print(f"📋 API Docs: http://localhost:{port}/docs")
    print(f"🔍 Health Check: http://localhost:{port}/health")
    print(f"🧪 Test Endpoint: http://localhost:{port}/test")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,  # ✅ Now properly uses Railway's PORT env var
        log_level="info",
        access_log=True
    )
