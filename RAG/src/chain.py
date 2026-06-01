from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.llm import get_llm
from src.prompt import get_prompt
from src.utils import format_docs
from src.history import get_session_history

def build_chain():
    retriever = get_retriever()
    llm       = get_llm()
    prompt    = get_prompt()

    rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    # Wrap with history
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )