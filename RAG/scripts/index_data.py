from src.vectorstore import get_embeddings
from src.config import settings
from langchain_qdrant import QdrantVectorStore

from langchain_community.document_loaders import CSVLoader

CSV_PATH = "data/mental_health_preprocessed.csv"

def index_data(force_recreate: bool = False):
    print("Loading documents ...")
    loader = CSVLoader(
        file_path=CSV_PATH,
        encoding="utf-8",
        source_column="Context",
        metadata_columns=["Response"],
    )
    docs = loader.load()
    docs = [d for d in docs if d.page_content.strip()]
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        force_recreate=force_recreate,
        batch_size=128,
        timeout=120.0,
    )
    print(f"{len(docs):,} documents loaded.\n")

if __name__ == "__main__":
    index_data(force_recreate=True)
