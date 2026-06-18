import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from RAG.src.config import settings

class EmotionClassifier:

    EMOTION_LABELS = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    def __init__(self, model_dir: str) -> None:
        model_path = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 128

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs      = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted     = torch.argmax(probabilities, dim=1).item()
            confidence    = probabilities[0, predicted].item()

        return {
            "emotion":    self.EMOTION_LABELS[predicted],
            "confidence": confidence,
        }


# ── load once at import time ──────────────────────────────────────────────────
_classifier = EmotionClassifier(settings.emotion_model_dir)

def get_emotion_classifier() -> EmotionClassifier:
    return _classifier 