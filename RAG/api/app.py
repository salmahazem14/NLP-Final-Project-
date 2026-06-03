from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import sys
import os

# Add parent directories to path so we can import src and modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # Add RAG/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))  # Add modules/ to path

from src.chain import build_chain
from src.history import delete_session_history
from translation_pipeline.translation_pipeline import translate  # type: ignore
from intent_classification.intent_classification import safe_classify_intent  # type: ignore
from emotion_classifier.emotion_classifier import EmotionClassifier  # type: ignore
from language_detection.language_detector import LanguageIdentifier  # type: ignore
from src.config import settings

app = FastAPI(title="Mental Health RAG API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chain = build_chain()

# Initialize classifiers
language_identifier = LanguageIdentifier()
language_identifier.load_model(settings.lang_model_path)
emotion_classifier = EmotionClassifier()
emotion_classifier.load_model(settings.emotion_model_dir)

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    k: int = 4

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    detected_language: str
    emotion: str
    intent: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    """
    Main chat endpoint that integrates all NLP modules:
    1. Language Detection
    2. Emotion Classification
    3. Intent Classification
    4. Translation (if needed)
    5. RAG-based Q&A
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        user_input = request.question
        
        # Step 1: Detect language
        detected_language = language_identifier.predict(user_input)
        
        # Step 2: Translate to English if needed
        english_text = user_input
        if detected_language.lower() not in ['en', 'english']:
            english_text = translate(user_input, detected_language, 'en')
        
        # Step 3: Detect emotion (on English text)
        emotion = emotion_classifier.predict(english_text, return_confidence=False)
        
        # Step 4: Classify intent (on English text)
        intent = safe_classify_intent(english_text)
        
        # Step 5: Get RAG response (always in English)
        rag_answer = chain.invoke(
            {"question": english_text},
            config={
                "configurable": {
                    "session_id": session_id,
                    "search_kwargs": {"k": request.k},
                }
            },
        )
        
        # Step 6: Translate response back to user's language if needed
        if detected_language.lower() not in ['en', 'english']:
            final_answer = translate(rag_answer, 'en', detected_language)
        else:
            final_answer = rag_answer
        
        return QueryResponse(
            answer=final_answer,
            session_id=session_id,
            detected_language=detected_language,
            emotion=emotion,
            intent=intent
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
def clear_history(session_id: str):
    delete_session_history(session_id)
    return {"status": "cleared", "session_id": session_id}