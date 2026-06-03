from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"Client: {doc.page_content}\nCounselor: {doc.metadata.get('Response', '')}"
        for doc in docs
    )