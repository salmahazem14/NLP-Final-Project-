from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chain import build_chain
from src.history import delete_session_history
import uuid

app   = FastAPI(title="Mental Health RAG API")
chain = build_chain()

class QueryRequest(BaseModel):
    question:   str
    session_id: str | None = None
    k:          int = 4

class QueryResponse(BaseModel):
    answer: str
    session_id:str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        print("here")
        answer = chain.invoke(
            {"question": request.question},
            config={
                "configurable": {
                    "session_id":    session_id,
                    "search_kwargs": {"k": request.k},
                }
            },
        )
        print(answer)
        return QueryResponse(answer=answer, session_id=session_id)
    except Exception as e:
        print(f"here *** {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
def clear_history(session_id: str):
    delete_session_history(session_id)
    return {"status": "cleared", "session_id": session_id}