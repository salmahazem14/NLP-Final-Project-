"""
Main Orchestrator for Mental Health Support Chatbot
Integrates all modules: Language Detection, Emotion Classification, Intent Classification, 
Translation Pipeline, and RAG-based Q&A
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add modules to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RAG'))

from modules.language_detection.language_detector import LanguageIdentifier
from modules.emotion_classifier.emotion_classifier import EmotionClassifier
from modules.intent_classification.intent_classification import safe_classify_intent
from modules.translation_pipeline.translation_pipeline import translate
from RAG.src.config import settings


class ChatbotPipeline:
    """Main orchestrator for the mental health chatbot."""
    
    def __init__(self):
        """Initialize all components."""
        print("Initializing Chatbot Pipeline...")
        
        # Language detection
        self.language_detector = LanguageIdentifier()
        self.language_detector.load_model(settings.lang_model_path)
        
        # Emotion classification
        self.emotion_classifier = EmotionClassifier()
        self.emotion_classifier.load_model(settings.emotion_model_dir)
        
        print("✓ All modules loaded successfully")
    
    def detect_language(self, text):
        """Detect the language of input text."""
        return self.language_detector.predict(text)
    
    def detect_emotion(self, text):
        """Detect the emotion in input text."""
        return self.emotion_classifier.predict(text, return_confidence=False)
    
    def classify_intent(self, text):
        """Classify the intent of input text."""
        return safe_classify_intent(text)
    
    def process_input(self, user_input):
        """
        Process user input through language detection and translation.
        
        Args:
            user_input (str): User's input in any language
            
        Returns:
            dict: {
                'original_text': str,
                'detected_language': str,
                'english_text': str,
                'emotion': str,
                'intent': str
            }
        """
        # Step 1: Detect language
        detected_language = self.detect_language(user_input)
        
        # Step 2: Translate to English if needed
        english_text = user_input
        if detected_language.lower() not in ['en', 'english']:
            english_text = translate(user_input, detected_language, 'en')
        
        # Step 3: Detect emotion (on English text for consistency)
        emotion = self.detect_emotion(english_text)
        
        # Step 4: Classify intent (on English text)
        intent = self.classify_intent(english_text)
        
        return {
            'original_text': user_input,
            'detected_language': detected_language,
            'english_text': english_text,
            'emotion': emotion,
            'intent': intent
        }
    
    def process_output(self, response_text, original_language):
        """
        Process RAG response and translate back to user's language if needed.
        
        Args:
            response_text (str): Response from RAG in English
            original_language (str): User's original language
            
        Returns:
            str: Response in user's original language
        """
        if original_language.lower() in ['en', 'english']:
            return response_text
        
        return translate(response_text, 'en', original_language)


def main():
    """Example usage of the chatbot pipeline."""
    pipeline = ChatbotPipeline()
    
    # Example: Process user input
    user_input = "I'm feeling anxious about my upcoming presentation"
    
    print(f"\n{'='*60}")
    print(f"User Input: {user_input}")
    print(f"{'='*60}")
    
    # Process input through all classifiers
    processed = pipeline.process_input(user_input)
    
    print(f"\nDetected Language: {processed['detected_language']}")
    print(f"English Text: {processed['english_text']}")
    print(f"Detected Emotion: {processed['emotion']}")
    print(f"Classified Intent: {processed['intent']}")
    
    # Example: Process output (translate back)
    rag_response = "Anxiety about presentations is common. Try these techniques: deep breathing, positive self-talk, and gradual exposure."
    
    final_response = pipeline.process_output(rag_response, processed['detected_language'])
    print(f"\nFinal Response: {final_response}")


if __name__ == "__main__":
    main()
