from pydantic_settings import BaseSettings

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
    lang_model_path: str = "../models/language_classifier_multi.pkl"
    emotion_model_dir: str = "../models/emotion_classifier"
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()