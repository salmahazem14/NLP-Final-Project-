from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]
def delete_session_history(session_id: str) -> bool:
    if session_id in _store:
        _store.pop(session_id, None)
    return True