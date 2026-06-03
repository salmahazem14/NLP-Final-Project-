# Mental Health Support Chatbot

A smart, end-to-end RAG-based chatbot that understands what you're feeling and helps with mental health questions. It auto-detects your language, recognizes your emotions, and gives you thoughtful answers in your own language.

## System Overview

Complete AI pipeline that processes user input through multiple specialized NLP modules:

```
User Input (Any Language)
    ↓
Language Detection → Identify language
    ↓
Translation Pipeline → Convert to English (if needed)
    ↓
Emotion Classifier → Detect emotional state
Intent Classifier → Determine response type
    ↓
    ├─→ If asking_mental_health_question:
    │      RAG Pipeline → Search knowledge base & generate response
    │
    └─→ If greeting, goodbye, gratitude, or out_of_scope:
           Direct Response → Use LLM directly (no RAG)
    ↓
Translation Pipeline → Convert back to user's language
    ↓
User Response (In their original language)
```

## What This Chatbot Does

1. **Understands Your Language** - Auto-detects language and responds in the same language
2. **Reads Your Emotions** - Figures out how you're feeling and responds with empathy
3. **Knows What You're Asking** - Classifies your intent to route to the best response method
4. **Smart Routing**:
   - Mental health questions → Uses RAG (searches knowledge base)
   - Greetings, goodbyes, thanks → Direct response (uses LLM only)
   - Out of scope → Polite redirection
5. **Translates Seamlessly** - Handles multilingual conversations automatically

## Core Modules

| Module | Technology | Purpose |
|--------|-----------|---------|
| **Language Detection** | TF-IDF + Logistic Regression | Identify input language |
| **Emotion Classifier** | DistilBERT (Fine-tuned) | Detect emotional state |
| **Intent Classifier** | Groq LLM (Few-shot) | Classify user intent with examples |
| **Translation Pipeline** | Groq LLM | Multilingual translation |
| **RAG Pipeline** | LangChain + Qdrant + Groq | Knowledge-grounded Q&A |

## Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (free: https://console.groq.com)
- Qdrant Cloud account (free: https://cloud.qdrant.io)

### Quick Start

**1. Setup:**
```bash
git clone <repo>
cd NLP-Final-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure:**
Create `.env` file:
```
# LLM & Vector DB
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

# Optional: For advanced features
HF_TOKEN=your_huggingface_token (for model downloads)
LANGSMITH_API_KEY=your_langsmith_key (for debugging)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_PROJECT=mental-health-rag
```

**3. Run Backend (Terminal 1):**
```bash
cd RAG/api
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**4. Run Frontend (Terminal 2):**
```bash
cd frontend
python -m http.server 5000
```

**5. Open browser:**
- Frontend: http://localhost:5000
- API: http://localhost:8000/health
- Configure frontend settings to use `http://localhost:8000`

## API Documentation

### POST `/chat` - Send message

**How it works:**
1. Detects language & emotion
2. Classifies intent
3. **If mental health question**: Uses RAG (searches knowledge base)
4. **If greeting/goodbye/thanks**: Direct LLM response (no search needed)
5. **Otherwise**: Polite redirection
6. Translates response back to user's language

**Request:**
```json
{
  "question": "I'm feeling anxious",
  "session_id": "optional_session_id",
  "k": 4
}
```

**Response:**
```json
{
  "answer": "Response text...",
  "session_id": "session_id",
  "detected_language": "en",
  "emotion": "fear",
  "intent": "asking_mental_health_question"
}
```

**Intent values:**
- `asking_mental_health_question` - Uses RAG for knowledge-grounded response
- `greeting` - Direct response (e.g., "Hello! How can I help?")
- `goodbye` - Direct response
- `gratitude` - Direct response
- `out_of_scope` - Polite redirect to mental health topics

### DELETE `/chat/{session_id}` - Clear history

Clears conversation history for a session.

### GET `/health` - Health check

Returns `{"status": "ok"}`

## Project Structure

```
NLP-Final-Project/
├── main.py                    # Integrated pipeline (standalone demo)
├── requirements.txt           # Dependencies
├── .env                      # Environment config (create this)
├── README.md                 # This file
│
├── modules/                  # Standalone NLP components
│   ├── language_detection/
│   ├── emotion_classifier/
│   ├── intent_classification/
│   └── translation_pipeline/
│
├── RAG/                      # RAG-based Q&A system
│   ├── api/app.py           # FastAPI backend (main entry point)
│   ├── src/                 # RAG implementation
│   └── Notebooks/           # Training notebooks
│
├── notebooks/                # Development notebooks
│   └── intent_classification/
│
└── frontend/                 # Web UI
    ├── index.html
    ├── app.js               # API integration
    ├── style.css
    └── README.md
```

## Technologies

- **Backend**: FastAPI, LangChain
- **Models**: DistilBERT, Sentence Transformers
- **APIs**: Groq, Qdrant
- **Frontend**: HTML, CSS, JavaScript
- **Databases**: Qdrant (vector)


