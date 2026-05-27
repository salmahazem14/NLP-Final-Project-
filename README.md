# Mental Health Support Chatbot

A smart chatbot that understands what you're feeling and helps you with mental health questions. It listens to what you say, recognizes your emotions, and gives you thoughtful answers.

## What This Chatbot Does

1. **Understands Your Language** - Detects what language you're speaking so it can respond in the same language
2. **Reads Your Emotions** - Figures out how you're feeling (happy, sad, anxious, etc.) and responds with empathy
3. **Knows What You're Asking** - Determines if you need a greeting, saying goodbye, asking for help, or something else
4. **Finds Answers for You** - Searches a mental health knowledge base and gives you personalized, helpful responses

## How It's Built

### 4 Main Modules

**1. Language Detection**
- Recognizes which language you're using
- Uses traditional NLP techniques
- Helps find the right answers in our database

**2. Emotion Classifier**
- Understands your emotional state
- Uses neural networks to recognize feelings
- Helps the chatbot respond with appropriate empathy

**3. Intent Classifier**
- Figures out what type of question you're asking
- Recognizes: greetings, goodbyes, thanks, mental health questions, or general queries
- Routes your question to the best answer method

**4. Q&A RAG Pipeline**
- Searches mental health counseling data for answers
- Uses smart embeddings to find the most relevant information
- Powered by AI to give you natural, helpful responses

**5. Translation Pipeline** (For Multilingual Support)
- Translates non-English user input to English for processing
- Translates RAG responses back to the user

## How the Pipeline Works

```
User Input (any language)
    ↓
Language Detector (identifies language)
    ↓
If language ≠ English:
  Translation Pipeline → English
    ↓
Intent Classifier (works on English)
Emotion Classifier (works on English)
    ↓
RAG Pipeline (searches in English, responds in English)
    ↓
If user's language ≠ English:
  Translation Pipeline → User's Language
    ↓
User Response (in original language)
```

## Getting Started

### Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

### Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Chatbot

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`

### Using the Translation Pipeline

The translation module handles converting between languages. It's called automatically when needed:

```python
from translation_pipeline import translate

# Translate Arabic to English
english_text = translate("أنا قلق", "ar", "en")
# Returns: "I am anxious"

# Translate English to French
french_text = translate("I feel stressed", "en", "fr")
# Returns: "Je me sens stressé"
```

In the full pipeline, the system:
1. Detects user's language using the language detector
2. If not English, translates input to English
3. Processes in English (intent, emotion, RAG)
4. If user's language wasn't English, translates response back

**Note**: Only non-English users trigger translation API calls.

## Technologies Used

- **NLP Models**: Transformers, Neural Networks
- **Embeddings**: Sentence Transformers
- **Vector Database**: Qdrant
- **LLM**: Groq API
- **Web Framework**: FastAPI
- **Frontend**: HTML, CSS, JavaScript

## Datasets

- Language Identification Dataset
- Emotion Classification Dataset
- Mental Health Counseling Conversations

---


