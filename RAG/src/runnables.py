from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from typing import Optional
from RAG.src.language_detector import get_detector
from langsmith import traceable
from RAG.src.emotion_classifier import get_emotion_classifier
from RAG.src.intent_classification import safe_classify_intent


class LanguageDetectorRunnable(RunnableSerializable):
    name: str = "LanguageDetectorRunnable"  
    @traceable(name="language_detector")         
    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> dict:
        lang = get_detector().predict(input["question"])
        return {**input, "lang": lang}
    
class EmotionClassifierRunnable(RunnableSerializable):
    name: str = "emotion_classifier"  
    @traceable(name="detect_emotion")         
    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> dict:
        result = get_emotion_classifier().predict(input["question"])
        return {**input, "emotion": result["emotion"], "emotion_confidence": result["confidence"]}
    

class IntentClassifierRunnable(RunnableSerializable):
    name: str = "intent_classifier"
    @traceable(name="intent_classifier")         
    def invoke(self,input: dict, config: Optional[RunnableConfig] = None) -> dict:
        intent = safe_classify_intent(input["question"])
        return {**input,"intent": intent,}