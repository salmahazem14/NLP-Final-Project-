"""Language Identification Module.

This module provides a production-ready interface for training, evaluating, 
saving, and invoking a TF-IDF & Logistic Regression language identification pipeline.
It supports both multi-language training and sub-language filtering (e.g., Arabic/English only).

Author: 
Date: May 2026
Project: RAG-Based Mental Health Support Chatbot - First module
"""

import logging
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Configure basic logging to help teammates trace the execution path
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LanguageIdentifier:
    """Manages the lifecycle of the Language Identification machine learning pipeline."""

    def __init__(self, max_iter: int = 2000, min_df: int = 2) -> None:
        """Initializes the Sklearn Pipeline with robust default hyperparameters.
        
        Using character n-grams (1 to 5) makes this model incredibly resilient against
        typos and highly effective across diverse alphabets (e.g., Arabic script vs. Latin script).
        """
        logging.info("Initializing LanguageIdentifier Pipeline architecture...")
        self.model: Pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 5),
                    lowercase=True,
                    strip_accents=None,
                    min_df=min_df
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    n_jobs=-1  # Utilize all available CPU cores for speed
                )
            )
        ])

    @staticmethod
    def clean_text(text: Union[str, float]) -> str:
        """Standardizes input strings to ensure uniform feature extraction.
        
        - Forces string type casting (handles unexpected NaNs gracefully).
        - Converts characters to lowercase.
        - Strips erratic whitespace/newlines.
        """
        text = str(text).lower()
        return " ".join(text.split())

    def prepare_data(self, dataset_name: str = "papluca/language-identification", 
                     languages: Optional[List[str]] = None) -> tuple:
        """Fetches the dataset from Hugging Face, cleans it, and filters targets if requested.

        Args:
            dataset_name: Hugging Face dataset registry path.
            languages: List of language codes (e.g., ['ar', 'en']) to filter by. 
                       If None, defaults to all 20 available languages.

        Returns:
            A tuple of splits: (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        logging.info(f"Loading dataset from Hugging Face: {dataset_name}...")
        dataset = load_dataset(dataset_name)
        
        # Convert raw dataset splits cleanly to Pandas DataFrames
        train_df = dataset["train"].to_pandas()
        valid_df = dataset["validation"].to_pandas()
        test_df = dataset["test"].to_pandas()

        # Apply targeted language filtering if your teammates only want a subset (e.g., Arabic & English)
        if languages:
            logging.info(f"Filtering dataset strictly for target languages: {languages}")
            train_df = train_df[train_df["labels"].isin(languages)]
            valid_df = valid_df[valid_df["labels"].isin(languages)]
            test_df = test_df[test_df["labels"].isin(languages)]

        # Run text cleaning routines across all data splits
        logging.info("Running text normalization and whitespace stripping...")
        for df in [train_df, valid_df, test_df]:
            df["text"] = df["text"].apply(self.clean_text)

        # Splitting features (X) and labels (y) explicitly to avoid reference bugs
        return (
            train_df["text"], train_df["labels"],
            valid_df["text"], valid_df["labels"],
            test_df["text"], test_df["labels"]
        )

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Fits the internal TF-IDF Vectorizer and Logistic Regression classifier."""
        logging.info(f"Commencing model training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        logging.info("Model training successfully completed!")

    def evaluate(self, X_eval: pd.Series, y_eval: pd.Series, split_name: str = "Validation") -> float:
        """Evaluates pipeline performance and outputs performance metrics.
        
        Prints a basic accuracy score along with a detailed classification report 
        (Precision, Recall, F1-Score) for deep errors analysis.
        """
        logging.info(f"Evaluating model performance on {split_name} split...")
        predictions = self.model.predict(X_eval)
        accuracy = accuracy_score(y_eval, predictions)
        
        print(f"\n================ {split_name.upper()} PERFORMANCE ================")
        print(f"Overall Accuracy: {accuracy:.4f}\n")
        print("Detailed Classification Metrics:")
        print(classification_report(y_eval, predictions))
        print("==================================================\n")
        
        return accuracy

    def save_model(self, file_path: str) -> None:
        """Serializes the entire pipeline object to disk using joblib."""
        logging.info(f"Saving serialized pipeline object to: {file_path}")
        joblib.dump(self.model, file_path)

    def load_model(self, file_path: str) -> None:
        """Loads a pre-trained serialized pipeline artifact from disk."""
        logging.info(f"Loading pipeline artifact from: {file_path}")
        self.model = joblib.load(file_path)

    def predict(self, text: str) -> str:
        """Infers the language code for a single string input.
        
        Cleans the string before passing it down to make sure formatting anomalies 
        don't tank the model's accuracy.
        """
        cleaned = self.clean_text(text)
        prediction = self.model.predict([cleaned])[0]
        return prediction
