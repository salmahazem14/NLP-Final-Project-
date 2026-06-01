from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )