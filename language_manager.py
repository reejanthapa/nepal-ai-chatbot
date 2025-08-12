# language_manager.py
"""
Language Detection and Management System for Nepal AI Chatbot
Handles automatic language detection, user preferences, and dynamic language switching
"""

import re
import logging
from typing import Dict, Any, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)

class Language(Enum):
    """Supported languages"""
    NEPALI = "nepali"
    ENGLISH = "english"
    AUTO = "auto"

class LanguageManager:
    """
    Advanced language management system with automatic detection and user preferences
    """
    
    def __init__(self):
        self.current_language = Language.AUTO
        self.user_preference = None
        self.detection_history = []
        
        # Language patterns for detection
        self.nepali_patterns = {
            'common_words': [
                'के', 'छ', 'हो', 'छैन', 'छु', 'छौं', 'छन्', 'थियो', 'थिए', 'हुन्छ',
                'गर्छु', 'गर्छौं', 'गर्छन्', 'भन्छ', 'भन्छन्', 'जान्छ', 'आउँछ',
                'खान्छ', 'पिउँछ', 'सुन्छ', 'हेर्छ', 'पढ्छ', 'लेख्छ', 'बोल्छ',
                'काम', 'घर', 'पानी', 'खाना', 'मान्छे', 'बाटो', 'गाउँ', 'शहर'
            ],
            'question_words': ['के', 'कसरी', 'कहाँ', 'कहिले', 'किन', 'कुन', 'को', 'कति'],
            'pronouns': ['म', 'तपाईं', 'उनी', 'हामी', 'तिमी', 'ऊ', 'उनीहरू'],
            'particles': ['ले', 'लाई', 'बाट', 'मा', 'देखि', 'सम्म', 'को', 'का', 'की'],
            'devanagari_script': r'[\u0900-\u097F]'
        }
        
        self.english_patterns = {
            'common_words': [
                'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does',
                'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must',
                'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when'
            ],
            'question_words': ['what', 'how', 'where', 'when', 'why', 'who', 'which'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her'],
            'articles': ['a', 'an', 'the'],
            'latin_script': r'[a-zA-Z]'
        }
        
        # Language switching commands
        self.language_switch_commands = {
            'to_nepali': [
                'switch to nepali', 'change to nepali', 'nepali language',
                'नेपालीमा', 'नेपाली भाषामा', 'नेपालीमा बोल', 'नेपाली भाषा',
                'नेपालीमा जवाफ दे', 'नेपालीमा भन'
            ],
            'to_english': [
                'switch to english', 'change to english', 'english language',
                'अंग्रेजीमा', 'अंग्रेजी भाषामा', 'अंग्रेजीमा बोल', 'अंग्रेजी भाषा',
                'अंग्रेजीमा जवाफ दे', 'इंग्लिशमा', 'english मा'
            ]
        }
    
    def detect_language(self, text: str) -> Tuple[Language, float]:
        """
        Detect language of input text with confidence score
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        if not text or not text.strip():
            return Language.ENGLISH, 0.0
        
        text_lower = text.lower().strip()
        
        # Check for Devanagari script (strongest indicator)
        devanagari_chars = len(re.findall(self.nepali_patterns['devanagari_script'], text))
        total_chars = len(re.sub(r'\s+', '', text))
        
        if total_chars > 0:
            devanagari_ratio = devanagari_chars / total_chars
            if devanagari_ratio > 0.3:  # If more than 30% Devanagari characters
                return Language.NEPALI, min(0.8 + devanagari_ratio * 0.2, 1.0)
        
        # Analyze words for language patterns
        words = text_lower.split()
        if not words:
            return Language.ENGLISH, 0.5
        
        nepali_score = self._calculate_language_score(words, self.nepali_patterns)
        english_score = self._calculate_language_score(words, self.english_patterns)
        
        # Normalize scores
        total_score = nepali_score + english_score
        if total_score > 0:
            nepali_confidence = nepali_score / total_score
            english_confidence = english_score / total_score
        else:
            nepali_confidence = 0.0
            english_confidence = 0.5  # Default to English if no patterns match
        
        # Determine language based on higher confidence
        if nepali_confidence > english_confidence:
            detected_language = Language.NEPALI
            confidence = nepali_confidence
        else:
            detected_language = Language.ENGLISH
            confidence = english_confidence
        
        # Store detection history
        self.detection_history.append({
            'text': text[:50] + '...' if len(text) > 50 else text,
            'detected_language': detected_language,
            'confidence': confidence,
            'nepali_score': nepali_score,
            'english_score': english_score
        })
        
        # Keep only recent history
        if len(self.detection_history) > 20:
            self.detection_history = self.detection_history[-10:]
        
        logger.info(f"Language detection: {detected_language.value} (confidence: {confidence:.2f})")
        return detected_language, confidence
    
    def _calculate_language_score(self, words: List[str], patterns: Dict[str, List[str]]) -> float:
        """Calculate language score based on pattern matching"""
        score = 0.0
        
        for word in words:
            # Check common words (high weight)
            if word in patterns.get('common_words', []):
                score += 3.0
            
            # Check question words (medium weight)
            if word in patterns.get('question_words', []):
                score += 2.0
            
            # Check pronouns (medium weight)
            if word in patterns.get('pronouns', []):
                score += 2.0
            
            # Check articles (for English) or particles (for Nepali)
            if word in patterns.get('articles', []) or word in patterns.get('particles', []):
                score += 1.5
        
        return score
    
    def check_language_switch_command(self, text: str) -> Tuple[bool, Language]:
        """
        Check if user wants to switch language
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_switch_command, target_language)
        """
        text_lower = text.lower().strip()
        
        # Check for Nepali switch commands
        for command in self.language_switch_commands['to_nepali']:
            if command in text_lower:
                return True, Language.NEPALI
        
        # Check for English switch commands
        for command in self.language_switch_commands['to_english']:
            if command in text_lower:
                return True, Language.ENGLISH
        
        return False, None
    
    def set_user_preference(self, language: Language):
        """Set user's preferred language"""
        self.user_preference = language
        self.current_language = language
        logger.info(f"User language preference set to: {language.value}")
    
    def get_response_language(self, input_text: str = None) -> Language:
        """
        Determine what language to use for response
        
        Args:
            input_text: Optional input text for auto-detection
            
        Returns:
            Language to use for response
        """
        # Check for language switch command first
        if input_text:
            is_switch, target_lang = self.check_language_switch_command(input_text)
            if is_switch:
                self.set_user_preference(target_lang)
                return target_lang
        
        # Use user preference if set
        if self.user_preference and self.user_preference != Language.AUTO:
            return self.user_preference
        
        # Auto-detect if preference is AUTO or not set
        if input_text:
            detected_lang, confidence = self.detect_language(input_text)
            if confidence > 0.6:  # High confidence threshold
                return detected_lang
        
        # Default to English if uncertain
        return Language.ENGLISH
    
    def get_language_selection_message(self) -> Dict[str, str]:
        """Get language selection message in both languages"""
        return {
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
    
    def parse_language_selection(self, user_input: str) -> Language:
        """
        Parse user's language selection
        
        Args:
            user_input: User's selection input
            
        Returns:
            Selected language
        """
        user_input = user_input.lower().strip()
        
        # English selections
        if user_input in ['english', 'eng', 'en', '1', 'e']:
            return Language.ENGLISH
        
        # Nepali selections  
        if user_input in ['nepali', 'नेपाली', 'nepal', 'np', '2', 'n']:
            return Language.NEPALI
        
        # Auto-detect based on input
        detected_lang, confidence = self.detect_language(user_input)
        if confidence > 0.7:
            return detected_lang
        
        # Default to English if unclear
        return Language.ENGLISH
    
    def get_language_switch_confirmation(self, new_language: Language) -> Dict[str, str]:
        """Get confirmation message for language switch"""
        if new_language == Language.NEPALI:
            return {
                'message': "✅ भाषा नेपालीमा बदलियो। अब म तपाईंलाई नेपालीमा जवाफ दिनेछु।",
                'english_message': "✅ Language switched to Nepali. I will now respond in Nepali."
            }
        else:
            return {
                'message': "✅ Language switched to English. I will now respond in English.",
                'nepali_message': "✅ भाषा अंग्रेजीमा बदलियो। अब म तपाईंलाई अंग्रेजीमा जवाफ दिनेछु।"
            }
    
    def get_current_language_info(self) -> Dict[str, Any]:
        """Get current language configuration info"""
        return {
            'current_language': self.current_language.value if self.current_language else 'auto',
            'user_preference': self.user_preference.value if self.user_preference else None,
            'detection_history_count': len(self.detection_history),
            'recent_detections': self.detection_history[-5:] if self.detection_history else []
        }
    
    def reset_language_preference(self):
        """Reset language preference to auto-detection"""
        self.user_preference = Language.AUTO
        self.current_language = Language.AUTO
        logger.info("Language preference reset to auto-detection")
    
    def is_language_related_query(self, text: str) -> bool:
        """Check if the query is about language switching or preferences"""
        language_keywords = [
            'language', 'भाषा', 'switch', 'change', 'बदल', 'छनोट',
            'english', 'nepali', 'अंग्रेजी', 'नेपाली'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in language_keywords)