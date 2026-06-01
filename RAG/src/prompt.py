from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a compassionate mental health counselor assistant.
Use the following real counselor responses as guidance only — do not copy them verbatim.
Respond with empathy and evidence-based support.
If the situation seems like a crisis, always recommend professional help or a crisis hotline.

Similar counseling examples:
{context}"""

def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])