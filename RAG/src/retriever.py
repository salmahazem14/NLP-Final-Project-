from langchain_core.runnables import ConfigurableField
from src.vectorstore import get_vector_store
from src.config import settings

def get_retriever():
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    ).configurable_fields(
        search_kwargs=ConfigurableField(id="search_kwargs")
    )