from cmath import log


from config.logging import get_logger, get_pipeline_logger, setup_logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import sys
import os

# Add parent directories to path so we can import src and modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # Add RAG/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))  # Add modules/ to path

from RAG.src.feedback import record_feedback
from RAG.src.chain import build_chain
from RAG.src.history import delete_session_history
from RAG.src.language_detector import get_detector
from RAG.src.emotion_classifier import get_emotion_classifier
from RAG.src.intent_classification import safe_classify_intent
from modules.translation_pipeline.translation_pipeline import translate  # type: ignore

from RAG.src.config import settings

# ── Telemetry ─────────────────────────────────────────────────────────────────
from .otel_metrics import record_intent, record_message_length, record_request
# ─────────────────────────────────────────────────────────────────────────────

setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("ENV", "development") == "production",
    log_to_file=True,
)
logger = get_logger(__name__)

app = FastAPI(title="Mental Health RAG API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware: record every request for server metric ────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    record_request(endpoint=request.url.path, status_code=response.status_code)
    return response
# ─────────────────────────────────────────────────────────────────────────────

logger.info("Building RAG chain...")
chain = build_chain()


# Initialize classifiers
logger.info("Loading language identifier...", extra={"model_path": settings.lang_model_path})
language_identifier = get_detector()

logger.info("Loading emotion classifier...", extra={"model_dir": settings.emotion_model_dir})
emotion_classifier = get_emotion_classifier()


logger.info("All models loaded. API ready.")

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

class FeedbackRequest(BaseModel):
    vote: str
    user_message: str
    bot_response: str

@app.get("/health")
def health():
    logger.debug("Health check called")
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
        log = get_pipeline_logger(logger, session_id=session_id)
        user_input = request.question

        # ── DATA metric: message length ───────────────────────────────────────
        record_message_length(user_input)
        # ─────────────────────────────────────────────────────────────────────

        # Step 1: Detect language
        detected_language = language_identifier.predict(user_input)
        log.info("Language detected", extra={"detected_language": detected_language})

        # Step 2: Translate to English if needed
        english_text = user_input
        if detected_language.lower() not in ['en', 'english']:
            log.debug("Translating input to English", extra={"from_lang": detected_language})
            english_text = translate(user_input, detected_language, 'en')
            log.debug("Translation complete")

        # Step 3: Detect emotion (on English text)
        emotion = emotion_classifier.predict(english_text)["emotion"]
        log.info("Emotion classified", extra={"emotion": emotion})

        # Step 4: Classify intent (on English text)
        intent = safe_classify_intent(english_text)
        log.info("Intent classified", extra={"intent": intent})

        # ── NLP metric: intent + emotion + language ───────────────────────────
        record_intent(intent=intent, emotion=emotion, language=detected_language)
        # ─────────────────────────────────────────────────────────────────────

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

        log.debug("RAG chain returned answer")

        # Step 6: Translate response back to user's language if needed
        if detected_language.lower() not in ['en', 'english']:
            log.debug("Translating response back", extra={"to_lang": detected_language})
            final_answer = translate(rag_answer, 'en', detected_language)
        else:
            final_answer = rag_answer

        log.info("Chat request completed successfully")

        return QueryResponse(
            answer=final_answer,
            session_id=session_id,
            detected_language=detected_language,
            emotion=emotion,
            intent=intent
        )
    except Exception as e:
        logger.error(
            "Chat request failed",
            exc_info=True,
            extra={"error": str(e)},
        )
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
def clear_history(session_id: str):
    delete_session_history(session_id)
    logger.info("Session history cleared")
    return {"status": "cleared", "session_id": session_id}


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        logger.info(
            "Feedback received",
            extra={
                "vote": request.vote
            }
        )

        result = record_feedback(request.vote)

        logger.info(
            "Feedback recorded successfully",
            extra=result
        )

        return result

    except Exception as e:
        logger.exception(
            "Failed to save feedback",
            extra={
                "vote": request.vote
            }
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save feedback"
        )