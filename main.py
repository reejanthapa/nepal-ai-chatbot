# api_main.py
"""
Nepal AI Chatbot - API Only Version (No Streamlit)
For Android Studio integration via FastAPI
"""

import google.generativeai as genai
import logging
import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from config import Config
from vector_engine import VectorEmbeddingEngine
from context_manager import ContextManager
from prompt_engineer import PromptEngineer
from data_loader import DatasetLoader
from language_manager import LanguageManager, Language

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NepalRAGChatbot:
    """
    Main RAG Chatbot class for API-only deployment
    Implements professional-grade RAG pipeline for Nepal domain with Google Gemini
    """
    
    def __init__(self):
        self.config = Config()
        self.data_loader = None
        self.vector_engine = None
        self.context_manager = None
        self.prompt_engineer = None
        self.language_manager = LanguageManager()
        self.gemini_model = None
        self.dataset = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time': 0.0,
            'average_confidence': 0.0,
            'cache_hits': 0,
            'language_switches': 0,
            'nepali_queries': 0,
            'english_queries': 0
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Nepal RAG Chatbot system with Google Gemini and Language Management...")
            
            # Initialize Gemini client
            self._setup_gemini_client()
            
            # Load and process dataset
            self._load_dataset()
            
            # Initialize RAG components
            self._initialize_rag_components()
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise
    
    def _setup_gemini_client(self):
        """Setup Google Gemini client with error handling"""
        try:
            api_key = Config.GEMINI_API_KEY
            
            if not api_key:
                raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in config.py")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model with safety settings
            generation_config = {
                "temperature": Config.TEMPERATURE,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": Config.MAX_TOKENS,
                "response_mime_type": "text/plain",
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            self.gemini_model = genai.GenerativeModel(
                model_name=Config.GEMINI_MODEL,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Test connection
            test_response = self.gemini_model.generate_content("Hello")
            logger.info("Gemini client initialized and tested successfully")
            
        except Exception as e:
            logger.error(f"Gemini client setup failed: {str(e)}")
            raise
    
    def _load_dataset(self):
        """Load and validate dataset"""
        try:
            self.data_loader = DatasetLoader(Config.DATASET_PATH)
            self.dataset = self.data_loader.load_dataset()
            
            # Display dataset info
            dataset_info = self.data_loader.get_dataset_info()
            logger.info(f"Loaded dataset with {len(self.dataset)} entries")
            
            # Show validation warnings if any
            if dataset_info['validation_errors']:
                logger.warning(f"Dataset has {len(dataset_info['validation_errors'])} validation issues")
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {str(e)}")
            raise
    
    def _initialize_rag_components(self):
        """Initialize RAG pipeline components"""
        try:
            # Extract questions for embedding
            questions = [item['input'] for item in self.dataset if 'input' in item]
            
            # Initialize vector engine
            self.vector_engine = VectorEmbeddingEngine(Config.EMBEDDING_METHOD)
            self.vector_engine.fit_transform(questions)
            
            # Initialize context manager
            self.context_manager = ContextManager(Config.MAX_CONTEXT_LENGTH)
            
            # Initialize prompt engineer
            self.prompt_engineer = PromptEngineer()
            
            logger.info("RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"RAG component initialization failed: {str(e)}")
            raise
    
    def process_query(self, user_question: str) -> Dict[str, Any]:
        """
        Process user query through complete RAG pipeline with language management
        
        Args:
            user_question: User's input question
            
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Update metrics
            self.performance_metrics['total_queries'] += 1
            
            # Step 0: Language Management
            # Check if this is a language switch command
            is_switch_command, target_lang = self.language_manager.check_language_switch_command(user_question)
            if is_switch_command:
                self.language_manager.set_user_preference(target_lang)
                self.performance_metrics['language_switches'] += 1
                return self._handle_language_switch_response(target_lang, start_time)
            
            # Determine response language
            response_language = self.language_manager.get_response_language(user_question)
            
            # Update language metrics
            if response_language == Language.NEPALI:
                self.performance_metrics['nepali_queries'] += 1
            else:
                self.performance_metrics['english_queries'] += 1
            
            # Step 1: Retrieval - Find relevant examples
            relevant_examples = self._retrieve_relevant_context(user_question)
            
            if not relevant_examples:
                return self._handle_no_context_response(user_question, response_language, start_time)
            
            # Step 2: Context Management - Select and format best examples
            selected_examples = self.context_manager.select_relevant_examples(
                relevant_examples, user_question, Config.MAX_EXAMPLES
            )
            
            # Step 3: Prompt Engineering - Select strategy and build prompt
            prompt_strategy = self.prompt_engineer.select_prompt_strategy(
                user_question, selected_examples
            )
            
            context_string = self.context_manager.build_context_string(
                selected_examples, user_question
            )
            
            # Build language-aware prompt for Gemini
            full_prompt = self.prompt_engineer.construct_gemini_prompt(
                user_question, context_string, prompt_strategy, response_language
            )
            
            # Step 4: Generation - Generate response using Gemini
            response = self._generate_gemini_response(full_prompt)
            
            # Step 5: Post-processing and metrics
            result = self._create_response_result(
                response, selected_examples, prompt_strategy, response_language, start_time
            )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return self._handle_error_response(user_question, str(e), start_time)
    
    def _handle_language_switch_response(self, target_lang: Language, start_time: float) -> Dict[str, Any]:
        """Handle language switch command"""
        confirmation = self.language_manager.get_language_switch_confirmation(target_lang)
        
        return {
            'response': confirmation['message'],
            'confidence': 1.0,
            'response_time': time.time() - start_time,
            'sources_used': 0,
            'prompt_strategy': 'language_switch',
            'relevant_examples': [],
            'language_switch': True,
            'new_language': target_lang.value,
            'metadata': {
                'type': 'language_switch',
                'target_language': target_lang.value
            }
        }
    
    def _retrieve_relevant_context(self, user_question: str) -> list:
        """Retrieve relevant examples using vector similarity"""
        try:
            # Transform query to embedding space
            query_embedding = self.vector_engine.transform(user_question)
            
            # Find similar examples
            similar_indices = self.vector_engine.compute_similarity(
                query_embedding, Config.TOP_K_RETRIEVAL
            )
            
            # Retrieve full examples with metadata
            relevant_examples = []
            for idx, similarity_score in similar_indices:
                if 0 <= idx < len(self.dataset):
                    example = self.dataset[idx].copy()
                    example['similarity'] = similarity_score
                    example['retrieval_rank'] = len(relevant_examples) + 1
                    relevant_examples.append(example)
            
            logger.debug(f"Retrieved {len(relevant_examples)} relevant examples")
            return relevant_examples
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return []
    
    def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Google Gemini"""
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Check if response was blocked by safety filters
            if not response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason.name == "SAFETY":
                        return "माफ गर्नुहोस्, तर म त्यो प्रश्नको जवाफ दिन सक्दिन। कृपया नेपाल सम्बन्धी अर्को प्रश्न सोध्नुहोस्। / I apologize, but I cannot provide a response to that query due to safety considerations. Please try rephrasing your question about Nepal."
                    else:
                        return "माफ गर्नुहोस्, तर मैले पूरा जवाफ दिन सकिन। कृपया आफ्नो प्रश्न फेरि भन्नुहोस्। / I apologize, but I couldn't generate a complete response. Please try rephrasing your question."
                else:
                    return "माफ गर्नुहोस्, तर म जवाफ दिन सकिन। कृपया नेपाल सम्बन्धी विषयहरूको बारेमा सोध्नुहोस्। / I apologize, but I couldn't generate a response. Please try asking about Nepal-related topics."
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise
    
    def _create_response_result(self, response_text: str, examples: list, 
                             strategy: str, language: Language, start_time: float) -> Dict[str, Any]:
        """Create comprehensive response result"""
        response_time = time.time() - start_time
        
        # Calculate confidence based on similarity scores and response quality
        if examples:
            avg_similarity = sum(ex.get('similarity', 0) for ex in examples) / len(examples)
            confidence = min(avg_similarity * 1.5, 1.0)  # Scale and cap at 1.0
        else:
            confidence = 0.2
        
        return {
            'response': response_text,
            'confidence': confidence,
            'response_time': response_time,
            'sources_used': len(examples),
            'prompt_strategy': strategy,
            'response_language': language.value,
            'relevant_examples': examples[:2],  # Include top 2 for debugging
            'metadata': {
                'model': Config.GEMINI_MODEL,
                'embedding_method': Config.EMBEDDING_METHOD,
                'retrieval_count': len(examples),
                'language_used': language.value
            }
        }
    
    def _handle_no_context_response(self, question: str, language: Language, start_time: float) -> Dict[str, Any]:
        """Handle case where no relevant context is found"""
        if language == Language.NEPALI:
            response = f"""मसँग त्यो विषयको बारेमा मेरो नेपाल ज्ञान आधारमा विशेष जानकारी छैन।

तथापि, म तपाईंलाई यी विषयहरूमा सहायता गर्न सक्छु:
🏔️ नेपाल घुमफिर र ट्रेकिङ
🎭 नेपाली संस्कृति र चाडपर्वहरू  
🗣️ नेपाली भाषा अनुवाद
🍲 खानेकुरा र नेपालमा भोजन
🏥 दैनिक जीवनका लागि व्यावहारिक जानकारी

कृपया आफ्नो प्रश्न अर्को तरिकाले सोध्नुहोस् वा नेपाल सम्बन्धी अर्को कुरा सोध्नुहोस्?"""
        else:
            response = f"""I don't have specific information about that topic in my Nepal knowledge base. 

However, I'd be happy to help you with other questions about:
🏔️ Nepal travel and trekking
🎭 Nepali culture and festivals
🗣️ Nepali language translation
🍲 Food and dining in Nepal
🏥 Practical information for daily life

Could you try rephrasing your question or ask about something else related to Nepal?"""
        
        return {
            'response': response,
            'confidence': 0.1,
            'response_time': time.time() - start_time,
            'sources_used': 0,
            'prompt_strategy': 'fallback',
            'response_language': language.value,
            'relevant_examples': [],
            'no_context': True
        }
    
    def _handle_error_response(self, question: str, error: str, start_time: float) -> Dict[str, Any]:
        """Handle error cases with bilingual support"""
        self.performance_metrics['failed_responses'] += 1
        
        # Get current language preference
        current_lang = self.language_manager.get_response_language()
        
        if current_lang == Language.NEPALI:
            error_message = f"""माफ गर्नुहोस्, तर तपाईंको प्रश्न प्रक्रिया गर्दा त्रुटि भयो। कृपया फेरि प्रयास गर्नुहोस् वा आफ्नो प्रश्न अर्को तरिकाले भन्नुहोस्।

यदि समस्या जारी रह्यो भने, तपाईं यी बारे सोध्न सक्नुहुन्छ:
• नेपाल यात्रा जानकारी
• सांस्कृतिक अभ्यास र चाडपर्वहरू
• भाषा अनुवाद
• नेपालको सामान्य जानकारी"""
        else:
            error_message = f"""I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.

If the problem persists, you can ask about:
• Nepal travel information
• Cultural practices and festivals
• Language translation
• General information about Nepal"""
        
        return {
            'response': error_message,
            'confidence': 0.0,
            'response_time': time.time() - start_time,
            'sources_used': 0,
            'prompt_strategy': 'error',
            'response_language': current_lang.value,
            'relevant_examples': [],
            'error': True,
            'error_message': error
        }
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance tracking metrics"""
        if not result.get('error', False):
            self.performance_metrics['successful_responses'] += 1
            
            # Update running averages
            total_success = self.performance_metrics['successful_responses']
            
            # Update average response time
            old_avg_time = self.performance_metrics['average_response_time']
            new_time = result['response_time']
            self.performance_metrics['average_response_time'] = (
                (old_avg_time * (total_success - 1) + new_time) / total_success
            )
            
            # Update average confidence
            old_avg_conf = self.performance_metrics['average_confidence']
            new_conf = result['confidence']
            self.performance_metrics['average_confidence'] = (
                (old_avg_conf * (total_success - 1) + new_conf) / total_success
            )
            
            # Update prompt strategy performance
            if hasattr(self.prompt_engineer, 'update_strategy_performance'):
                self.prompt_engineer.update_strategy_performance(
                    result['prompt_strategy'], result['confidence']
                )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        base_metrics = self.performance_metrics.copy()
        
        # Calculate derived metrics
        total_queries = base_metrics['total_queries']
        if total_queries > 0:
            base_metrics['success_rate'] = (
                base_metrics['successful_responses'] / total_queries
            ) * 100
            base_metrics['error_rate'] = (
                base_metrics['failed_responses'] / total_queries
            ) * 100
            base_metrics['nepali_percentage'] = (
                base_metrics['nepali_queries'] / total_queries
            ) * 100
            base_metrics['english_percentage'] = (
                base_metrics['english_queries'] / total_queries
            ) * 100
        else:
            base_metrics['success_rate'] = 0.0
            base_metrics['error_rate'] = 0.0
            base_metrics['nepali_percentage'] = 0.0
            base_metrics['english_percentage'] = 0.0
        
        # Add system info
        base_metrics.update({
            'dataset_size': len(self.dataset),
            'embedding_method': Config.EMBEDDING_METHOD,
            'model': Config.GEMINI_MODEL,
            'current_language': self.language_manager.current_language.value,
            'user_preference': self.language_manager.user_preference.value if self.language_manager.user_preference else 'auto',
            'system_initialized': all([
                self.vector_engine is not None,
                self.context_manager is not None,
                self.prompt_engineer is not None,
                self.gemini_model is not None,
                self.language_manager is not None
            ])
        })
        
        return base_metrics