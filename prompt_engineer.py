# prompt_engineer.py
"""
Advanced Prompt Engineering for Nepal AI Assistant
Optimized for bilingual (Nepali-English) responses for Nepal locals and tourists
"""

from typing import List, Dict, Any, Tuple
import re
import logging
from config import Config, PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class PromptEngineer:
    """
    Advanced prompt engineering system optimized for Nepal-focused bilingual responses
    """
    
    def __init__(self):
        self.prompt_templates = PROMPT_TEMPLATES
        self.strategy_patterns = self._initialize_strategy_patterns()
        self.prompt_history = []
        self.performance_metrics = {
            'default': {'uses': 0, 'avg_confidence': 0.0},
            'cultural': {'uses': 0, 'avg_confidence': 0.0},
            'translation': {'uses': 0, 'avg_confidence': 0.0},
            'travel': {'uses': 0, 'avg_confidence': 0.0}
        }
    
    def _initialize_strategy_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for prompt strategy selection (including Nepali patterns)"""
        return {
            'translation': [
                r'\btranslate\b', r'\bnepali\b', r'\bhow to say\b', r'\bwhat.*mean\b',
                r'\blanguage\b', r'\bspeak\b', r'\bword for\b', r'\bpronounce\b',
                r'\bdevanagari\b', r'\bromanized\b', r'अनुवाद', r'भन्न', r'अर्थ',
                r'भाषा', r'बोल्न', r'शब्द'
            ],
            'cultural': [
                r'\bfestival\b', r'\btradition\b', r'\bculture\b', r'\bcustom\b',
                r'\bcelebrate\b', r'\britual\b', r'\breligion\b', r'\bethnicity\b',
                r'\bdashain\b', r'\btihar\b', r'\bholi\b', r'\bteej\b',
                r'\bwedding\b', r'\bceremony\b', r'\bpractice\b',
                r'चाड', r'पर्व', r'संस्कृति', r'परम्परा', r'मनाउन', r'दशैं',
                r'तिहार', r'होली', r'तीज', r'विवाह', r'समारोह'
            ],
            'travel': [
                r'\btravel\b', r'\btrek\b', r'\bvisit\b', r'\bgo to\b', r'\btransport\b',
                r'\bhotel\b', r'\baccommodation\b', r'\bpermit\b', r'\bvisa\b',
                r'\bflight\b', r'\bbus\b', r'\btaxi\b', r'\beverest\b', r'\bannapurna\b',
                r'\bchitwan\b', r'\bpokhara\b', r'\bkathmandu\b', r'\bthamel\b',
                r'\bbackpack\b', r'\bguide\b', r'\bsafety\b',
                r'घुम्न', r'यात्रा', r'जान', r'होटल', r'बास', r'अनुमति',
                r'भिसा', r'बस', r'ट्याक्सी', r'एभरेस्ट', r'अन्नपूर्ण',
                r'चितवन', r'पोखरा', r'काठमाडौं', r'थमेल'
            ]
        }
    
    def select_prompt_strategy(self, user_question: str, examples: List[Dict] = None) -> str:
        """Select the best prompt strategy based on question analysis"""
        question_lower = user_question.lower()
        strategy_scores = {'default': 0.1}  # Base score for default
        
        # Score based on pattern matching
        for strategy, patterns in self.strategy_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, question_lower))
                score += matches
            
            if score > 0:
                strategy_scores[strategy] = score
        
        # Boost scores based on example content if available
        if examples:
            self._boost_scores_from_examples(strategy_scores, examples)
        
        # Consider historical performance
        self._apply_performance_weighting(strategy_scores)
        
        # Select strategy with highest score
        selected_strategy = max(strategy_scores, key=strategy_scores.get)
        
        logger.info(f"Selected prompt strategy: {selected_strategy} (scores: {strategy_scores})")
        return selected_strategy
    
    def _boost_scores_from_examples(self, scores: Dict[str, float], examples: List[Dict]):
        """Boost strategy scores based on retrieved examples content"""
        for example in examples[:2]:  # Only consider top 2 examples
            example_text = (example.get('input', '') + ' ' + 
                          example.get('output', '') + ' ' +
                          example.get('nepali', '')).lower()
            
            # Cultural indicators
            if any(word in example_text for word in ['festival', 'tradition', 'culture', 'custom', 'चाड', 'संस्कृति']):
                scores['cultural'] = scores.get('cultural', 0) + 0.5
            
            # Translation indicators
            if any(word in example_text for word in ['translate', 'nepali', 'language', 'अनुवाद', 'भाषा']):
                scores['translation'] = scores.get('translation', 0) + 0.5
            
            # Travel indicators
            if any(word in example_text for word in ['trek', 'travel', 'visit', 'transport', 'घुम्न', 'यात्रा']):
                scores['travel'] = scores.get('travel', 0) + 0.5
    
    def _apply_performance_weighting(self, scores: Dict[str, float]):
        """Apply historical performance weighting to strategy selection"""
        for strategy in scores:
            if strategy in self.performance_metrics:
                metrics = self.performance_metrics[strategy]
                if metrics['uses'] > 0:
                    performance_boost = metrics['avg_confidence'] * 0.2
                    scores[strategy] += performance_boost
    
    def construct_gemini_prompt(self, 
                               user_question: str, 
                               context: str, 
                               strategy: str = 'default',
                               response_language = None) -> str:
        """
        Construct optimized bilingual prompt for Google Gemini API
        
        Args:
            user_question: User's input question
            context: Retrieved context from knowledge base
            strategy: Selected prompt strategy
            response_language: Target language for response (from Language enum)
            
        Returns:
            Single prompt string for Gemini API optimized for Nepal context
        """
        # Import here to avoid circular import
        from language_manager import Language
        
        # Get system prompt for strategy
        system_prompt = self.prompt_templates.get(strategy, self.prompt_templates['default'])
        
        # Enhance system prompt with strategy-specific instructions
        enhanced_system_prompt = self._enhance_system_prompt(system_prompt, strategy, user_question)
        
        # Construct comprehensive prompt for Gemini with language preference
        full_prompt = self._construct_nepal_focused_prompt(
            enhanced_system_prompt, user_question, context, strategy, response_language
        )
        
        # Store prompt in history for analysis
        self._store_prompt_history(user_question, strategy, full_prompt)
        
        return full_prompt
    
    def _construct_nepal_focused_prompt(self, 
                                       system_prompt: str, 
                                       user_question: str, 
                                       context: str, 
                                       strategy: str,
                                       response_language = None) -> str:
        """Construct Nepal-focused bilingual prompt with language preference"""
        
        # Import here to avoid circular import
        from language_manager import Language
        
        # Determine language instruction based on preference
        if response_language == Language.NEPALI:
            main_instruction = "CRITICAL: तपाईंको जवाफ मुख्यतः नेपाली भाषामा (देवनागरी लिपिमा) होस्।"
            language_note = "नेपालीमा"
        elif response_language == Language.ENGLISH:
            main_instruction = "CRITICAL: Please respond primarily in English language."
            language_note = "in English"
        else:
            # Auto-detect or default
            main_instruction = "CRITICAL: तपाईंको जवाफ मुख्यतः नेपाली भाषामा (देवनागरी लिपिमा) होस्।"
            language_note = "नेपालीमा"
        
        prompt_parts = [
            "=== नेपाल AI सहायक निर्देशन (Nepal AI Assistant Instructions) ===",
            system_prompt,
            "",
            "=== नेपाल ज्ञान आधार सन्दर्भ (Nepal Knowledge Base Context) ===",
            context,
            "",
            "=== प्रयोगकर्ताको प्रश्न (User Question) ===",
            f"प्रश्न: {user_question}",
            "",
            "=== जवाफ निर्देशनहरू (Response Guidelines) ===",
            main_instruction,
            "CRITICAL: प्रदान गरिएको ज्ञान आधारको सन्दर्भलाई प्राथमिकता दिनुहोस्।",
            ""
        ]
        
        # Add strategy-specific guidelines based on language preference
        if strategy == 'translation':
            if response_language == Language.ENGLISH:
                prompt_parts.extend([
                    "Translation Guidelines:",
                    "1. Provide accurate Nepali translation in Devanagari script",
                    "2. Include romanized pronunciation for English speakers", 
                    "3. Explain cultural context behind the language use",
                    "4. Offer alternative phrases for different formality levels"
                ])
            else:
                prompt_parts.extend([
                    "अनुवाद निर्देशनहरू (Translation Guidelines):",
                    "1. नेपाली अनुवाद देवनागरी लिपिमा प्रदान गर्नुहोस्",
                    "2. अंग्रेजी बोल्नेहरूका लागि रोमानाइज्ड उच्चारण दिनुहोस्",
                    "3. भाषा प्रयोगको सांस्कृतिक सन्दर्भ व्याख्या गर्नुहोस्",
                    "4. औपचारिक र अनौपचारिक रूपका विकल्पहरू प्रस्तुत गर्नुहोस्"
                ])
        elif strategy == 'cultural':
            if response_language == Language.ENGLISH:
                prompt_parts.extend([
                    "Cultural Guidelines:",
                    "1. Provide rich cultural context and historical significance",
                    "2. Include regional or ethnic variations if applicable",
                    "3. Explain both traditional and modern practices",
                    "4. Be respectful of cultural sensitivities and diverse practices"
                ])
            else:
                prompt_parts.extend([
                    "सांस्कृतिक निर्देशनहरू (Cultural Guidelines):",
                    "1. समृद्ध सांस्कृतिक सन्दर्भ र ऐतिहासिक महत्व प्रदान गर्नुहोस्",
                    "2. क्षेत्रीय वा जातीय भिन्नताहरू उल्लेख गर्नुहोस्",
                    "3. पारम्परिक र आधुनिक दुवै अभ्यासहरू व्याख्या गर्नुहोस्",
                    "4. सांस्कृतिक संवेदनशीलता र विविधताको सम्मान गर्नुहोस्"
                ])
        elif strategy == 'travel':
            if response_language == Language.ENGLISH:
                prompt_parts.extend([
                    "Travel Guidelines:",
                    "1. Include practical, actionable travel advice",
                    "2. Mention current costs, permits, and logistics where relevant",
                    "3. Provide safety considerations and tips",
                    "4. Include seasonal timing recommendations",
                    "5. Consider different budget levels (budget, mid-range, luxury)"
                ])
            else:
                prompt_parts.extend([
                    "यात्रा निर्देशनहरू (Travel Guidelines):",
                    "1. व्यावहारिक र कार्यान्वयनयोग्य यात्रा सल्लाह दिनुहोस्",
                    "2. हालको लागत, अनुमतिपत्र र रसदका बारेमा उल्लेख गर्नुहोस्",
                    "3. सुरक्षा विचार र सुझावहरू प्रदान गर्नुहोस्",
                    "4. मौसमी समय र बजेट स्तरहरू विचार गर्नुहोस्",
                    "5. महत्वपूर्ण अंग्रेजी शब्दहरू कोष्ठकमा राख्नुहोस्"
                ])
        else:  # default
            if response_language == Language.ENGLISH:
                prompt_parts.extend([
                    "General Guidelines:",
                    "1. Provide comprehensive, accurate information about Nepal",
                    "2. Include Nepali terms in parentheses when appropriate",
                    "3. Add practical tips and cultural context",
                    "4. Be helpful for both locals and tourists",
                    "5. Use the knowledge base context as your primary source"
                ])
            else:
                prompt_parts.extend([
                    "सामान्य निर्देशनहरू (General Guidelines):",
                    "1. नेपालको बारेमा व्यापक र सटीक जानकारी प्रदान गर्नुहोस्",
                    "2. आवश्यक परेमा अंग्रेजी शब्दहरू कोष्ठकमा राख्नुहोस्",
                    "3. व्यावहारिक सुझाव र सांस्कृतिक सन्दर्भ थप्नुहोस्",
                    "4. स्थानीय र पर्यटक दुवैका लागि उपयोगी होस्",
                    "5. ज्ञान आधारको सन्दर्भलाई प्राथमिक स्रोतको रूपमा प्रयोग गर्नुहोस्"
                ])
        
        # Final instructions based on language
        if response_language == Language.ENGLISH:
            prompt_parts.extend([
                "",
                "=== Important Reminders ===",
                "- Use the 'nepali' section from knowledge base when providing Nepali translations",
                "- Respond primarily in English as requested",
                "- If context doesn't contain relevant information, state so clearly",
                "- Don't make up information not in the knowledge base",
                "- Format your response clearly with proper structure",
                "",
                "Please provide your response in English now:"
            ])
        else:
            prompt_parts.extend([
                "",
                "=== महत्वपूर्ण सम्झनाहरू (Important Reminders) ===",
                "- ज्ञान आधारमा भएको 'nepali' खण्डको जानकारी प्रयोग गर्नुहोस्",
                "- जवाफ मुख्यतः नेपाली भाषामा दिनुहोस्",
                "- यदि सन्दर्भमा प्रासंगिक जानकारी छैन भने स्पष्ट रूपमा भन्नुहोस्",
                "- ज्ञान आधारमा नभएको जानकारी बनाउनुहुन्न",
                "- जवाफलाई स्पष्ट संरचनाका साथ ढाँचा दिनुहोस्",
                "",
                f"कृपया अब तपाईंको जवाफ {language_note} दिनुहोस्:"
            ])
        
        return "\n".join(prompt_parts)
    
    def _enhance_system_prompt(self, base_prompt: str, strategy: str, user_question: str) -> str:
        """Enhance system prompt with dynamic instructions"""
        enhancements = []
        
        # Detect if question is in Nepali or English
        nepali_indicators = ['के', 'कसरी', 'कहाँ', 'कहिले', 'किन', 'कुन', 'को']
        is_nepali_question = any(indicator in user_question for indicator in nepali_indicators)
        
        if is_nepali_question:
            enhancements.append("प्रश्न नेपालीमा छ। जवाफ पनि नेपालीमै दिनुहोस्।")
        
        # Add strategy-specific enhancements
        if strategy == 'translation':
            if 'formal' in user_question.lower() or 'आदर' in user_question:
                enhancements.append("औपचारिक/आदरसूचक भाषा रूपहरूमा ध्यान दिनुहोस्।")
            if 'pronunciation' in user_question.lower() or 'उच्चारण' in user_question:
                enhancements.append("विस्तृत उच्चारण निर्देशन समावेश गर्नुहोस्।")
        
        elif strategy == 'cultural':
            festivals = ['dashain', 'tihar', 'holi', 'दशैं', 'तिहार', 'होली']
            if any(festival in user_question.lower() for festival in festivals):
                enhancements.append("चाडपर्वको विस्तृत जानकारी मिति, महत्व र मनाउने तरिकासहित प्रदान गर्नुहोस्।")
        
        elif strategy == 'travel':
            if 'budget' in user_question.lower() or 'cost' in user_question.lower() or 'लागत' in user_question:
                enhancements.append("विस्तृत लागतको जानकारी र बजेट अनुकूल विकल्पहरू समावेश गर्नुहोस्।")
            if 'safety' in user_question.lower() or 'सुरक्षा' in user_question:
                enhancements.append("सुरक्षा जानकारी र वर्तमान सावधानीहरूलाई प्राथमिकता दिनुहोस्।")
        
        # Add question complexity analysis
        word_count = len(user_question.split())
        if word_count > 15:
            enhancements.append("यो जटिल बहु-भागीय प्रश्न हो। प्रत्येक पक्षलाई व्यवस्थित रूपमा सम्बोधन गर्नुहोस्।")
        
        # Combine base prompt with enhancements
        if enhancements:
            enhanced_prompt = f"{base_prompt}\n\nयो प्रश्नका लागि अतिरिक्त निर्देशनहरू:\n- " + "\n- ".join(enhancements)
            return enhanced_prompt
        
        return base_prompt
    
    def _store_prompt_history(self, question: str, strategy: str, full_prompt: str):
        """Store prompt in history for performance analysis"""
        self.prompt_history.append({
            'question': question,
            'strategy': strategy,
            'prompt': full_prompt,
            'timestamp': __import__('datetime').datetime.now()
        })
        
        # Keep only recent history to manage memory
        if len(self.prompt_history) > 100:
            self.prompt_history = self.prompt_history[-50:]
    
    def update_strategy_performance(self, strategy: str, confidence_score: float):
        """Update performance metrics for a strategy"""
        if strategy in self.performance_metrics:
            metrics = self.performance_metrics[strategy]
            
            # Update running average
            old_avg = metrics['avg_confidence']
            old_count = metrics['uses']
            new_count = old_count + 1
            
            new_avg = (old_avg * old_count + confidence_score) / new_count
            
            metrics['uses'] = new_count
            metrics['avg_confidence'] = new_avg
            
            logger.debug(f"Updated {strategy} strategy: uses={new_count}, avg_confidence={new_avg:.3f}")
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies"""
        return {
            'metrics': self.performance_metrics.copy(),
            'total_prompts': len(self.prompt_history),
            'strategy_distribution': self._get_strategy_distribution()
        }
    
    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategy usage"""
        distribution = {}
        for entry in self.prompt_history:
            strategy = entry['strategy']
            distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution
    
    def optimize_prompt_templates(self):
        """Optimize prompt templates based on performance data (future enhancement)"""
        # This is a placeholder for future ML-based prompt optimization
        # Could analyze which prompt variations lead to better responses
        logger.info("Prompt template optimization not yet implemented")
    
    def get_prompt_suggestions(self, user_question: str) -> List[str]:
        """Generate suggested follow-up questions or clarifications"""
        suggestions = []
        question_lower = user_question.lower()
        
        # Generic suggestions
        if len(user_question.split()) < 4:
            suggestions.append("Could you provide more details about what specifically you'd like to know?")
            suggestions.append("तपाईं के विशेष जान्न चाहनुहुन्छ त्यसको बारेमा थप विवरण दिन सक्नुहुन्छ?")
        
        # Strategy-specific suggestions
        if 'translate' in question_lower or 'अनुवाद' in question_lower:
            suggestions.extend([
                "Would you like both formal and informal versions?",
                "Do you need pronunciation guidance?",
                "Are you looking for a specific context (business, casual, etc.)?",
                "के तपाईंलाई औपचारिक र अनौपचारिक दुवै चाहिन्छ?",
                "उच्चारण निर्देशन चाहिन्छ?"
            ])
        
        elif any(word in question_lower for word in ['festival', 'culture', 'चाड', 'संस्कृति']):
            suggestions.extend([
                "Would you like to know about regional variations?",
                "Are you interested in the historical significance?",
                "Do you want to know how it's celebrated today?",
                "के तपाईं क्षेत्रीय भिन्नताहरूको बारेमा जान्न चाहनुहुन्छ?",
                "ऐतिहासिक महत्वमा रुचि छ?"
            ])
        
        elif any(word in question_lower for word in ['trek', 'travel', 'घुम्न', 'यात्रा']):
            suggestions.extend([
                "What's your experience level with trekking?",
                "What time of year are you planning to visit?",
                "Are you looking for budget or luxury options?",
                "तपाईंको ट्रेकिङ अनुभव कस्तो छ?",
                "कुन समयमा भ्रमण गर्ने योजना छ?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def analyze_question_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze question complexity for prompt optimization"""
        words = question.split()
        
        # Count Nepali and English question words
        english_q_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which']
        nepali_q_words = ['के', 'कहाँ', 'कहिले', 'किन', 'कसरी', 'को', 'कुन']
        
        english_questions = len([w for w in words if w.lower() in english_q_words])
        nepali_questions = len([w for w in words if w in nepali_q_words])
        
        analysis = {
            'word_count': len(words),
            'english_question_words': english_questions,
            'nepali_question_words': nepali_questions,
            'total_question_words': english_questions + nepali_questions,
            'complexity': 'simple',
            'language_detected': 'mixed',
            'multiple_topics': False,
            'requires_translation': 'translate' in question.lower() or 'अनुवाद' in question.lower(),
            'requires_cultural_context': any(word in question.lower() for word in ['culture', 'tradition', 'festival', 'custom', 'संस्कृति', 'परम्परा', 'चाड'])
        }
        
        # Detect primary language
        if nepali_questions > english_questions:
            analysis['language_detected'] = 'nepali'
        elif english_questions > nepali_questions:
            analysis['language_detected'] = 'english'
        elif any(char in question for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञ'):
            analysis['language_detected'] = 'nepali'
        else:
            analysis['language_detected'] = 'english'
        
        # Determine complexity
        total_q_words = analysis['total_question_words']
        if len(words) > 20 or total_q_words > 2:
            analysis['complexity'] = 'complex'
        elif len(words) > 10 or total_q_words > 1:
            analysis['complexity'] = 'medium'
        
        # Check for multiple topics
        topic_indicators = ['and', 'also', 'additionally', 'plus', 'moreover', 'र', 'पनि', 'साथै']
        analysis['multiple_topics'] = any(indicator in question.lower() for indicator in topic_indicators)
        
        return analysis
    
    def get_language_appropriate_response_template(self, language: str) -> str:
        """Get response template based on language preference"""
        if language == 'nepali':
            return """मुख्य जानकारी:
{main_content}

थप सुझावहरू:
{additional_tips}

सम्बन्धित विषयहरू:
{related_topics}"""
        else:
            return """Main Information:
{main_content}

Additional Tips:
{additional_tips}

Related Topics:
{related_topics}"""
    
    def construct_fallback_prompt(self, user_question: str, language: str) -> str:
        """Construct fallback prompt when no context is found"""
        if language == 'nepali':
            return f"""तपाईं नेपालका लागि AI सहायक हुनुहुन्छ। प्रयोगकर्ताले सोधेको प्रश्न: "{user_question}"

यो प्रश्नको उत्तर तपाईंको ज्ञान आधारमा छैन, तर तपाईं नेपालको सामान्य ज्ञानको आधारमा सहायक जवाफ दिन सक्नुहुन्छ।

कृपया:
1. माफी माग्नुहोस् कि विशेष जानकारी छैन
2. सामान्य सहायक जानकारी दिनुहोस् यदि सम्भव छ भने  
3. सम्बन्धित विषयहरूको सुझाव दिनुहोस्
4. प्रयोगकर्तालाई अन्य प्रश्न सोध्न प्रोत्साहन दिनुहोस्

जवाफ नेपालीमा दिनुहोस्।"""
        else:
            return f"""You are an AI assistant for Nepal. The user asked: "{user_question}"

This specific question is not in your knowledge base, but you can provide a helpful response based on general knowledge about Nepal.

Please:
1. Apologize that you don't have specific information
2. Provide general helpful information if possible
3. Suggest related topics they might ask about
4. Encourage them to ask other questions

Respond in English."""