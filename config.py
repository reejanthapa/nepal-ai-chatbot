# config.py
"""
Configuration settings for Nepal AI Chatbot
Updated for bilingual (Nepali-English) responses focused on Nepal locals and tourists
"""

import os
from typing import Dict, Any

class Config:
    """Centralized configuration management"""
    
   # API Configuration - Changed to Gemini
    GEMINI_API_KEY = "AIzaSyD2KRGzXbHS9A5dlX9Py2HzM_7iU_VD4KQ"  # Your free Gemini API key
    GEMINI_MODEL = "gemini-1.5-flash"  # Free tier model
    MAX_TOKENS = 600
    TEMPERATURE = 0.3
   
   
    
    # === Model Parameters ===
    TEMPERATURE = 0.3  # Lower for more consistent responses
    MAX_TOKENS = 1000
    TOP_P = 0.95
    TOP_K = 64
    
    # === Dataset Configuration ===
    DATASET_PATH = "Dataset.json"
    
    # === RAG Pipeline Settings ===
    EMBEDDING_METHOD = "hybrid"  # Better for multilingual content
    TOP_K_RETRIEVAL = 10
    MAX_EXAMPLES = 3  # Fewer examples for cleaner context
    MAX_CONTEXT_LENGTH = 2000  # Reduced for better focus
    MIN_SIMILARITY_THRESHOLD = 0.15  # Slightly higher threshold
    
    # === TF-IDF Settings ===
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    CHAR_NGRAM_RANGE = (2, 4)
    
    # === Performance Settings ===
    ENABLE_CACHING = True
    LOG_LEVEL = "INFO"
    
    # === Streamlit Settings ===
    PAGE_TITLE = "🇳🇵 Nepal AI सहायक - स्थानीय र पर्यटकहरूको लागि"
    PAGE_ICON = "🇳🇵"

# Enhanced prompt templates optimized for Nepal locals and tourists
PROMPT_TEMPLATES: Dict[str, str] = {
    'default': """तपाईं नेपालका लागि विशेष रूपमा डिजाइन गरिएको AI सहायक हुनुहुन्छ। तपाईंको मुख्य उद्देश्य स्थानीय नेपालीहरू र नेपाल भ्रमणमा आउने पर्यटकहरूलाई सहायता गर्नु हो।

आपकी भूमिका (Your Role):
- नेपालको बारेमा सटीक र उपयोगी जानकारी प्रदान गर्नुहोस्
- प्रदान गरिएको ज्ञान आधारलाई प्राथमिक स्रोतको रूपमा प्रयोग गर्नुहोस्
- सांस्कृतिक रूपमा संवेदनशील र सम्मानजनक हुनुहोस्
- स्थानीय र पर्यटक दुवैका लागि उपयोगी जानकारी प्रदान गर्नुहोस्

Response Guidelines:
- प्रयोगकर्ताको भाषा प्राथमिकता अनुसार जवाफ दिनुहोस्
- महत्वपूर्ण शब्दहरू दुवै भाषामा उल्लेख गर्नुहोस् जब उपयुक्त हो
- व्यावहारिक सुझावहरू समावेश गर्नुहोस्
- ज्ञान आधारमा भएका उदाहरणहरूको प्रयोग गर्नुहोस्

विशेष निर्देशन: प्रयोगकर्ताको भाषा प्राथमिकता र प्रश्नको भाषा अनुसार उत्तम भाषामा जवाफ दिनुहोस्।""",

    'translation': """तपाईं नेपाली-अंग्रेजी भाषा अनुवादमा विशेषज्ञ हुनुहुन्छ। तपाईंको विशेषज्ञता दुवै भाषा र संस्कृतिको गहिरो ज्ञान समावेश गर्दछ।

Your Translation Expertise (तपाईंको अनुवाद विशेषज्ञता):
- नेपाली (देवनागरी) र अंग्रेजी बीच सही अनुवाद
- उच्चारण निर्देशन र रोमानाइजेशन
- भाषा प्रयोगको सांस्कृतिक सन्दर्भ र बारीकी
- औपचारिक र अनौपचारिक भाषाको भिन्नता
- क्षेत्रीय बोली र भिन्नताहरू

Translation Guidelines:
1. सटीक अनुवाद प्रदान गर्नुहोस् (देवनागरी स्क्रिप्टमा यदि नेपालीमा)
2. अंग्रेजी बोल्नेहरूका लागि रोमानाइज्ड उच्चारण दिनुहोस्
3. सांस्कृतिक सन्दर्भ व्याख्या गर्नुहोस्
4. विभिन्न औपचारिकता स्तरका लागि विकल्पहरू प्रस्तुत गर्नुहोस्
5. भाषा प्रयोगको पछाडिको अर्थ र संकेत व्याख्या गर्नुहोस्""",

    'cultural': """तपाईं नेपालको संस्कृतिको विशेषज्ञ हुनुहुन्छ जसलाई परम्परा, चाडपर्व, रीतिरिवाज, र सामाजिक प्रथाहरूको व्यापक ज्ञान छ।

Cultural Expertise (सांस्कृतिक विशेषज्ञता):
- दशैं, तिहार, होली, तीज जस्ता मुख्य चाडपर्वहरू
- धार्मिक र आध्यात्मिक परम्पराहरू  
- पारम्परिक समारोह र अनुष्ठानहरू
- क्षेत्रीय र जातीय विविधताहरू
- ऐतिहासिक सन्दर्भ र आधुनिक अनुकूलन

Cultural Response Guidelines:
1. समृद्ध सांस्कृतिक सन्दर्भ र ऐतिहासिक महत्व प्रदान गर्नुहोस्
2. प्रासंगिक भएमा ऐतिहासिक पृष्ठभूमि समावेश गर्नुहोस्
3. क्षेत्रीय वा जातीय भिन्नताहरू उल्लेख गर्नुहोस्
4. पारम्परिक र समकालीन दुवै अभ्यासहरू व्याख्या गर्नुहोस्
5. सांस्कृतिक संवेदनशीलता र विविधताको सम्मान गर्नुहोस्
6. चाडपर्वहरूका मिति, महत्व, र मनाउने तरिकाहरू व्याख्या गर्नुहोस्""",

    'travel': """तपाईं नेपाल भ्रमणको विशेषज्ञ हुनुहुन्छ जसलाई पर्यटन, ट्रेकिङ, यातायात, र व्यावहारिक यात्रा जानकारीको व्यापक ज्ञान छ।

Travel Expertise (यात्रा विशेषज्ञता):
- ट्रेकिङ मार्गहरू र पहाडी पर्यटन
- यातायातका विकल्पहरू र रसद व्यवस्थापन
- बासको सिफारिसहरू (बजेट देखि लक्जरी सम्म)
- अनुमतिपत्र र भिसा आवश्यकताहरू
- सुरक्षा विचारहरू र सर्वोत्तम अभ्यासहरू
- मौसमी समय र मौसम विचारहरू
- स्थानीय संस्कृति र शिष्टाचार

Travel Response Guidelines:
1. व्यावहारिक र कार्यान्वयनयोग्य यात्रा सल्लाह दिनुहोस्
2. हालको लागत अनुमान सम्भव भएसम्म समावेश गर्नुहोस्
3. सुरक्षा विचार र सुझावहरू उल्लेख गर्नुहोस्
4. मौसमी समय सिफारिसहरू समावेश गर्नुहोस्
5. विभिन्न अनुभव स्तर र बजेटहरू विचार गर्नुहोस्
6. सांस्कृतिक संवेदनशीलता र स्थानीय शिष्टाचारको सल्लाह दिनुहोस्
7. महत्वपूर्ण नेपाली शब्दहरू र तिनका अर्थहरू समावेश गर्नुहोस्"""
}

# Example questions for the sidebar (bilingual)
EXAMPLE_QUESTIONS = [
    # Nepali Questions
    "नेपालीमा 'नमस्ते' कसरी भनिन्छ?",
    "दशैं चाड के हो र कसरी मनाइन्छ?",
    "काठमाडौंबाट पोखरा कसरी जाने?",
    "नेपालमा के खाने कुराहरू छन्?",
    "एभरेस्ट बेस क्याम्पका लागि अनुमति कसरी लिने?",
    
    # English Questions
    "How do you say 'Thank you' in Nepali?",
    "What is Tihar festival and how is it celebrated?", 
    "Best trekking routes for beginners in Nepal?",
    "What should I know about Nepali wedding customs?",
    "When is the best time to visit Nepal?",
    
    # Mixed/Travel
    "काठमाडौंका राम्रा अस्पतालहरू कहाँ छन्?",
    "How to get permits for Annapurna Circuit?",
    "नेपाली भाषा सिक्ने उत्तम तरिका के हो?",
    "What food should tourists try in Nepal?",
    "होली कहिले मनाइन्छ र के गर्छन्?"
]

# Language-specific welcome messages
WELCOME_MESSAGES = {
    'nepali': """🙏 **नमस्ते! नेपाल AI सहायकमा स्वागत छ!**

म एक उन्नत AI सहायक हुँ जो **RAG (Retrieval Augmented Generation)** प्रविधि र **Google Gemini** द्वारा संचालित छु, नेपाल-विशिष्ट ज्ञानमा विशेष रूपमा प्रशिक्षित।

**🔥 मेरो विशेषताहरू:**
- 🧠 **२५०+ व्यवस्थित उदाहरणहरू** नेपालको बारेमा  
- 🔍 **स्मार्ट सिमान्टिक खोज** प्रासंगिक जानकारी फेला पार्न
- 📝 **डायनामिक प्रोम्प्ट इन्जिनियरिङ** उत्तम जवाफका लागि
- 🎯 **सन्दर्भ-सजग प्रक्रिया** सटीक उत्तरका लागी
- 🌍 **द्विभाषिक समर्थन** (नेपाली/English)

**💬 म तपाईंलाई यी विषयहरूमा सहायता गर्न सक्छु:**
- 🏔️ घुमफिर र ट्रेकिङ जानकारी
- 🎭 सांस्कृतिक अभ्यास र चाडपर्वहरू  
- 🗣️ नेपाली भाषा अनुवाद
- 🍲 खानेकुरा र भोजन सिफारिसहरू
- 🏥 दैनिक जीवन र व्यावहारिक जानकारी

**नेपालको बारेमा जे पनि सोध्नुहोस्!** साइडबारमा भएका उदाहरण प्रश्नहरू पनि क्लिक गर्न सक्नुहुन्छ।""",

    'english': """🙏 **Namaste! Welcome to Nepal AI Assistant!**

I'm an advanced AI assistant powered by **RAG (Retrieval Augmented Generation)** technology and **Google Gemini**, specially trained on Nepal-specific knowledge.

**🔥 What makes me special:**
- 🧠 **250+ curated examples** about Nepal
- 🔍 **Smart semantic search** to find relevant information
- 📝 **Dynamic prompt engineering** for optimal responses  
- 🎯 **Context-aware processing** for accurate answers
- 🌍 **Bilingual support** (नेपाली/English)

**💬 I can help you with:**
- 🏔️ Travel & trekking information
- 🎭 Cultural practices & festivals
- 🗣️ Nepali language translation
- 🍲 Food & dining recommendations  
- 🏥 Daily life & practical information

**Ask me anything about Nepal!** You can also click the example questions in the sidebar."""
}

# Language detection patterns
LANGUAGE_PATTERNS = {
    'nepali': {
        'script_pattern': r'[\u0900-\u097F]',  # Devanagari Unicode range
        'common_words': [
            'के', 'छ', 'हो', 'छैन', 'छु', 'छौं', 'छन्', 'थियो', 'थिए', 'हुन्छ',
            'गर्छु', 'गर्छौं', 'गर्छन्', 'भन्छ', 'भन्छन्', 'जान्छ', 'आउँछ'
        ],
        'question_words': ['के', 'कसरी', 'कहाँ', 'कहिले', 'किन', 'कुन', 'को', 'कति'],
        'greetings': ['नमस्ते', 'नमस्कार', 'राम्रो', 'हजुर']
    },
    'english': {
        'common_words': [
            'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does',
            'will', 'would', 'can', 'could', 'should', 'may', 'might'
        ],
        'question_words': ['what', 'how', 'where', 'when', 'why', 'who', 'which'],
        'greetings': ['hello', 'hi', 'namaste', 'good']
    }
}

# Error messages in both languages
ERROR_MESSAGES = {
    'nepali': {
        'no_context': """मसँग त्यो विषयको बारेमा मेरो नेपाल ज्ञान आधारमा विशेष जानकारी छैन।

तथापि, म तपाईंलाई यी विषयहरूमा सहायता गर्न सक्छु:
🏔️ नेपाल घुमफिर र ट्रेकिङ
🎭 नेपाली संस्कृति र चाडपर्वहरू  
🗣️ नेपाली भाषा अनुवाद
🍲 खानेकुरा र नेपालमा भोजन
🏥 दैनिक जीवनका लागि व्यावहारिक जानकारी

कृपया आफ्नो प्रश्न अर्को तरिकाले सोध्नुहोस् वा नेपाल सम्बन्धी अर्को कुरा सोध्नुहोस्?""",
        
        'processing_error': """माफ गर्नुहोस्, तर तपाईंको प्रश्न प्रक्रिया गर्दा त्रुटि भयो। कृपया फेरि प्रयास गर्नुहोस् वा आफ्नो प्रश्न अर्को तरिकाले भन्नुहोस्।

यदि समस्या जारी रह्यो भने, तपाईं यी बारे सोध्न सक्नुहुन्छ:
• नेपाल यात्रा जानकारी
• सांस्कृतिक अभ्यास र चाडपर्वहरू
• भाषा अनुवाद
• नेपालको सामान्य जानकारी""",
        
        'safety_block': """माफ गर्नुहोस्, तर म त्यो प्रश्नको जवाफ दिन सक्दिन। कृपया नेपाल सम्बन्धी अर्को प्रश्न सोध्नुहोस्।"""
    },
    
    'english': {
        'no_context': """I don't have specific information about that topic in my Nepal knowledge base.

However, I'd be happy to help you with other questions about:
🏔️ Nepal travel and trekking
🎭 Nepali culture and festivals
🗣️ Nepali language translation
🍲 Food and dining in Nepal
🏥 Practical information for daily life

Could you try rephrasing your question or ask about something else related to Nepal?""",
        
        'processing_error': """I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.

If the problem persists, you can ask about:
• Nepal travel information
• Cultural practices and festivals
• Language translation
• General information about Nepal""",
        
        'safety_block': """I apologize, but I cannot provide a response to that query due to safety considerations. Please try asking about Nepal-related topics."""
    }
}

# Language switching commands
LANGUAGE_SWITCH_COMMANDS = {
    'to_nepali': [
        'switch to nepali', 'change to nepali', 'nepali language', 'respond in nepali',
        'नेपालीमा', 'नेपाली भाषामा', 'नेपालीमा बोल', 'नेपाली भाषा',
        'नेपालीमा जवाफ दे', 'नेपालीमा भन', 'नेपालीमा बदल'
    ],
    'to_english': [
        'switch to english', 'change to english', 'english language', 'respond in english',
        'अंग्रेजीमा', 'अंग्रेजी भाषामा', 'अंग्रेजीमा बोल', 'अंग्रेजी भाषा',
        'अंग्रेजीमा जवाफ दे', 'इंग्लिशमा', 'english मा', 'अंग्रेजीमा बदल'
    ]
}

# Language selection messages
LANGUAGE_SELECTION_MESSAGES = {
    'initial': {
        'english': """🌍 **Language Selection / भाषा छनोट**

Welcome to Nepal AI Assistant! I can help you in both Nepali and English.

Please choose your preferred language:
- Type **"English"** or **"1"** for English responses
- Type **"Nepali"** or **"नेपाली"** or **"2"** for Nepali responses

You can switch languages anytime by saying "switch to English" or "नेपालीमा बदल".""",
        
        'nepali': """🌍 **भाषा छनोट / Language Selection**

नेपाल AI सहायकमा स्वागत छ! म तपाईंलाई नेपाली र अंग्रेजी दुवै भाषामा सहायता गर्न सक्छु।

कृपया आफ्नो मनपर्ने भाषा छान्नुहोस्:
- अंग्रेजी जवाफका लागि **"English"** वा **"1"** टाइप गर्नुहोस्
- नेपाली जवाफका लागि **"Nepali"** वा **"नेपाली"** वा **"2"** टाइप गर्नुहोस्

तपाईं जुनसुकै बेला "switch to English" वा "नेपालीमा बदल" भनेर भाषा बदल्न सक्नुहुन्छ।"""
    }
}

# Confirmation messages for language switching
LANGUAGE_SWITCH_CONFIRMATIONS = {
    'to_nepali': {
        'message': "✅ भाषा नेपालीमा बदलियो। अब म तपाईंलाई नेपालीमा जवाफ दिनेछु।",
        'english_message': "✅ Language switched to Nepali. I will now respond in Nepali."
    },
    'to_english': {
        'message': "✅ Language switched to English. I will now respond in English.",
        'nepali_message': "✅ भाषा अंग्रेजीमा बदलियो। अब म तपाईंलाई अंग्रेजीमा जवाफ दिनेछु।"
    }
}

# Chat input placeholders
CHAT_PLACEHOLDERS = {
    'nepali': "नेपालको बारेमा जे पनि सोध्नुहोस्... 🇳🇵",
    'english': "Ask me anything about Nepal... 🇳🇵",
    'auto': "Ask me anything about Nepal / नेपालको बारेमा जे पनि सोध्नुहोस्... 🇳🇵"
}

# System status messages
SYSTEM_STATUS = {
    'initializing': {
        'nepali': '🚀 नेपाल AI सहायक सुरु गर्दै...',
        'english': '🚀 Initializing Nepal AI Assistant...'
    },
    'ready': {
        'nepali': '✅ सिस्टम सफलतापूर्वक सुरु भयो!',
        'english': '✅ System initialized successfully!'
    },
    'processing': {
        'nepali': '🧠 RAG pipeline + Gemini द्वारा प्रक्रिया गर्दै...',
        'english': '🧠 Processing with RAG pipeline + Gemini...'
    }
}

# Sidebar labels and headers
SIDEBAR_LABELS = {
    'metrics_header': {
        'nepali': '📊 सिस्टम मेट्रिक्स',
        'english': '📊 System Metrics'
    },
    'features_header': {
        'nepali': '🎯 उन्नत सुविधाहरू',
        'english': '🎯 Advanced Features'
    },
    'examples_header': {
        'nepali': '💡 उदाहरण प्रश्नहरू',
        'english': '💡 Example Questions'
    },
    'language_distribution': {
        'nepali': 'भाषा वितरण',
        'english': 'Language Distribution'
    },
    'current_language': {
        'nepali': 'हालको भाषा',
        'english': 'Current Language'
    }
}

# Metrics labels
METRICS_LABELS = {
    'total_queries': {
        'nepali': 'कुल प्रश्न',
        'english': 'Total Queries'
    },
    'success_rate': {
        'nepali': 'सफलता दर',
        'english': 'Success Rate'
    },
    'avg_time': {
        'nepali': 'औसत समय',
        'english': 'Avg Time'
    },
    'confidence': {
        'nepali': 'औसत विश्वास',
        'english': 'Avg Confidence'
    },
    'dataset_size': {
        'nepali': 'डेटासेट साइज',
        'english': 'Dataset Size'
    }
}

# Footer content
FOOTER_CONTENT = {
    'architecture': {
        'nepali': '🔧 वास्तुकला',
        'english': '🔧 Architecture'
    },
    'features': {
        'nepali': '📊 सुविधाहरू',
        'english': '📊 Features'
    },
    'clear_chat': {
        'nepali': '🗑️ Chat History साफ गर्नुहोस्',
        'english': '🗑️ Clear Chat History'
    },
    'credits': {
        'nepali': '🏗️ निर्माण / Built with: Streamlit • Google Gemini • scikit-learn • Advanced RAG Architecture<br>🇳🇵 नेपालका लागि विशेष • सांस्कृतिक रूपमा संवेदनशील • व्यावसायिक रूपमा इन्जिनियर गरिएको<br>🌍 द्विभाषिक समर्थन • स्वचालित भाषा पहिचान • Dynamic Language Switching',
        'english': '🏗️ Built with: Streamlit • Google Gemini • scikit-learn • Advanced RAG Architecture<br>🇳🇵 Specialized for Nepal • Culturally Sensitive • Professionally Engineered<br>🌍 Bilingual Support • Auto Language Detection • Dynamic Language Switching'
    }
}

# Advanced features list
ADVANCED_FEATURES = [
    "✅ RAG Architecture",
    "✅ Vector Embeddings", 
    "✅ Smart Context Management",
    "✅ Dynamic Prompt Engineering",
    "✅ Bilingual Support (नेपाली/English)",
    "✅ Auto Language Detection",
    "✅ Google Gemini Integration"
]

# Validation and quality thresholds
QUALITY_THRESHOLDS = {
    'MIN_CONFIDENCE': 0.3,
    'HIGH_CONFIDENCE': 0.7,
    'MIN_SIMILARITY': 0.15,
    'HIGH_SIMILARITY': 0.6,
    'MIN_RESPONSE_LENGTH': 10,
    'MAX_RESPONSE_LENGTH': 2000
}

# Debug and development settings
DEBUG_SETTINGS = {
    'SHOW_ANALYTICS': True,
    'LOG_PROMPTS': False,  # Set to True for debugging
    'TRACK_PERFORMANCE': True,
    'ENABLE_PROFILING': False
}