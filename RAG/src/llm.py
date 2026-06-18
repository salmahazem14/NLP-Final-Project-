from langchain_groq import ChatGroq
from RAG.src.config import settings

def get_llm():
    return ChatGroq(
        model=settings.final_llm, # TODO : Change to be in settings also
        temperature=0.7,
        max_tokens=1024,
        api_key=settings.groq_api_key,
    )