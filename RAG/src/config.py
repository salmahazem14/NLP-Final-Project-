from pydantic_settings import BaseSettings
import os

# Get the project root directory (3 levels up from this file)
# RAG/src/config.py -> RAG/src -> RAG -> Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    groq_api_key: str
    collection_name: str = "mental_health_counseling"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k: int = 4
    final_llm:str = "llama-3.3-70b-versatile"
    hf_token: str = ""
    lang_model_path: str = os.path.join(MODELS_DIR, "language_classifier/language_classifier_multi.pkl")
    emotion_model_dir: str = os.path.join(MODELS_DIR, "emotion_classifier")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()