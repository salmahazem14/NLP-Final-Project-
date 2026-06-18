import joblib
from typing import Union

class LanguageIdentifier:

    def __init__(self, model_path: str) -> None:
        self.model = joblib.load(model_path)

    @staticmethod
    def clean_text(text: Union[str, float]) -> str:
        text = str(text).lower()
        return " ".join(text.split())

    def predict(self, text: str) -> str:
        cleaned = self.clean_text(text)
        return self.model.predict([cleaned])[0]


# ── load once at import time ──────────────────────────────────────────────────
from RAG.src.config import settings
_detector = LanguageIdentifier(settings.lang_model_path)

def get_detector() -> LanguageIdentifier:
    return _detector