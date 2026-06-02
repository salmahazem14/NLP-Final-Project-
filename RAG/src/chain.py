from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.llm import get_llm
from src.prompt import get_prompt
from src.utils import format_docs
from src.history import get_session_history
from src.runnables import LanguageDetectorRunnable, EmotionClassifierRunnable, IntentClassifierRunnable
from langchain_core.runnables import RunnableBranch

# lang detection input+lang
# emotion classifier input+emotion(e.g. joy, fear, so on)
# intent classifier(LLM call) route to (RAG + LLM) or LLM only


def build_chain():
    retriever = get_retriever()
    llm       = get_llm()
    prompt    = get_prompt()


    direct_chain = (
        {
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
            "lang": RunnableLambda(lambda x: x.get("lang", "en")),
            "emotion": RunnableLambda(lambda x: x.get("emotion", "neutral")),
            "context": RunnableLambda(lambda _: ""),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question":RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
            "lang": RunnableLambda(lambda x: x.get("lang", "en")),
            "emotion": RunnableLambda(lambda x: x.get("emotion", "neutral")),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    router = RunnableBranch(
            (
                lambda x: x["intent"] == "asking_mental_health_question",
                rag_chain,
            ),
            direct_chain,
        )

    chain = (
        LanguageDetectorRunnable()
        | EmotionClassifierRunnable()
        | IntentClassifierRunnable()
        | router
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )