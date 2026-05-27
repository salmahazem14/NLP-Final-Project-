"""Emotion Classification Module using Transformer Architecture.

This module provides a production-ready emotion classifier fine-tuned on DistilBERT.
Designed specifically for mental health chatbot applications where understanding 
user emotional state is critical for generating appropriate, empathetic responses.

Key Features:
- Fine-tuned DistilBERT for 6-class emotion detection
- Confidence scoring for uncertainty detection
- Class imbalance handling with weighted loss
- Early stopping and learning rate scheduling
- Batch prediction for efficiency
- Model versioning and metadata tracking
- Comprehensive error handling and validation

Author: 
Date: May 2026
Project: RAG-Based Mental Health Support Chatbot - Emotion Detection Module
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Suppress unnecessary warnings in production
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmotionDataset(Dataset):
    """PyTorch Dataset wrapper for emotion classification data."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Args:
            texts: List of input text strings
            labels: List of integer emotion labels
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns tokenized text and label as tensors."""
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize with proper padding and truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class EmotionClassifier:
    """
    Production-ready emotion classifier using fine-tuned DistilBERT.
    
    Handles the complete ML lifecycle: data preparation, training with early stopping,
    evaluation with comprehensive metrics, model persistence, and inference with
    confidence scores.
    """

    # Emotion label mapping from the dair-ai/emotion dataset
    EMOTION_LABELS = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        """
        Initialize the emotion classifier with a pre-trained transformer model.

        Args:
            model_name: HuggingFace model identifier (default: DistilBERT)
            max_length: Maximum token sequence length
            device: Target device ('cuda', 'cpu', or None for auto-detection)
        """
        logging.info(f"Initializing EmotionClassifier with model: {model_name}")

        # Auto-detect optimal device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logging.info(f"Using device: {self.device}")

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.training_history = {"train_loss": [], "val_loss": [], "val_f1": []}

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logging.info("Tokenizer loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Tokenizer initialization failed: {e}") from e

    def prepare_data(
        self,
        dataset_name: str = "dair-ai/emotion",
        batch_size: int = 32,
        test_size: float = 0.1,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare the emotion dataset with proper batching.

        Args:
            dataset_name: HuggingFace dataset identifier
            batch_size: Training batch size
            test_size: Fraction of train data to use for validation
            num_workers: DataLoader workers (0 for containers, 2-4 for local GPU)

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logging.info(f"Loading dataset: {dataset_name}")

        try:
            dataset = load_dataset(dataset_name)
        except Exception as e:
            logging.error(f"Dataset loading failed: {e}")
            raise ValueError(f"Cannot load dataset '{dataset_name}': {e}") from e

        # Extract splits
        train_data = dataset["train"]
        test_data = dataset["test"]
        val_data = dataset["validation"]

        logging.info(
            f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        # Create PyTorch datasets
        train_dataset = EmotionDataset(
            texts=train_data["text"],
            labels=train_data["label"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        val_dataset = EmotionDataset(
            texts=val_data["text"],
            labels=val_data["label"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        test_dataset = EmotionDataset(
            texts=test_data["text"],
            labels=test_data["label"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Create data loaders with proper shuffling
        # Note: num_workers=0 avoids multiprocessing issues in containers/notebooks
        # For local development with GPUs, you can increase to num_workers=2-4
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        logging.info("Data loaders created successfully")
        return train_loader, val_loader, test_loader

    def _calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Calculate class weights to handle imbalanced emotion distribution.
        Critical for mental health apps where some emotions may be underrepresented.
        """
        logging.info("Calculating class weights for imbalanced data handling...")
        label_counts = torch.zeros(len(self.EMOTION_LABELS))

        for batch in train_loader:
            labels = batch["label"]
            for label in labels:
                label_counts[label] += 1

        # Inverse frequency weighting
        total_samples = label_counts.sum()
        class_weights = total_samples / (len(self.EMOTION_LABELS) * label_counts)

        logging.info(f"Class weights: {class_weights.tolist()}")
        return class_weights.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        early_stopping_patience: int = 3,
        use_class_weights: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the transformer model with advanced training techniques.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum training epochs
            learning_rate: Peak learning rate for AdamW
            warmup_steps: Linear warmup steps for learning rate
            early_stopping_patience: Epochs to wait before stopping if no improvement
            use_class_weights: Whether to apply class weighting for imbalance

        Returns:
            Training history dictionary with losses and metrics
        """
        logging.info("Initializing model for training...")

        # Initialize model
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(self.EMOTION_LABELS)
            )
            self.model.to(self.device)
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize model: {e}") from e

        # Calculate class weights if enabled
        class_weights = None
        if use_class_weights:
            class_weights = self._calculate_class_weights(train_loader)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # AdamW optimizer (better for transformers than standard Adam)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Early stopping tracking
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None

        logging.info(
            f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch"
        )

        for epoch in range(epochs):
            # ==================== TRAINING PHASE ====================
            self.model.train()
            train_loss = 0.0
            train_progress = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
            )

            for batch in train_progress:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = criterion(outputs.logits, labels)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_progress.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            self.training_history["train_loss"].append(avg_train_loss)

            # ==================== VALIDATION PHASE ====================
            val_loss, val_f1, val_metrics = self._evaluate_model(
                val_loader, criterion, split_name="Validation"
            )
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_f1"].append(val_f1)

            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )

            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logging.info(f"✓ New best validation F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                logging.info(
                    f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
                )

                if patience_counter >= early_stopping_patience:
                    logging.info("Early stopping triggered!")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logging.info("Restored best model from training")

        return self.training_history

    def _evaluate_model(
        self, data_loader: DataLoader, criterion: nn.Module, split_name: str = "Test"
    ) -> Tuple[float, float, Dict]:
        """
        Internal evaluation method with comprehensive metrics.

        Returns:
            Tuple of (loss, f1_score, metrics_dict)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Train or load a model first.")

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_predictions, average="weighted")

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1_weighted": f1,
            "predictions": all_predictions,
            "labels": all_labels,
        }

        return avg_loss, f1, metrics

    def evaluate(
        self, test_loader: DataLoader, show_confusion_matrix: bool = True
    ) -> Dict:
        """
        Evaluate model performance with detailed metrics.

        Args:
            test_loader: Test data loader
            show_confusion_matrix: Whether to compute and display confusion matrix

        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info("Evaluating model on test set...")

        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Cannot evaluate.")

        criterion = nn.CrossEntropyLoss()
        _, _, metrics = self._evaluate_model(test_loader, criterion, split_name="Test")

        # Print comprehensive report
        print("\n" + "=" * 70)
        print(f"{'TEST SET EVALUATION':^70}")
        print("=" * 70)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print("\nPer-Class Performance:")
        print("-" * 70)

        report = classification_report(
            metrics["labels"],
            metrics["predictions"],
            target_names=list(self.EMOTION_LABELS.values()),
            digits=4,
        )
        print(report)

        if show_confusion_matrix:
            cm = confusion_matrix(metrics["labels"], metrics["predictions"])
            print("\nConfusion Matrix:")
            print("-" * 70)
            print("Rows: True Labels | Columns: Predicted Labels")
            print(cm)

        print("=" * 70 + "\n")

        return metrics

    def predict(
        self, text: str, return_confidence: bool = True
    ) -> Union[str, Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict emotion for a single text input.

        Args:
            text: Input text string
            return_confidence: If True, return confidence scores for all emotions

        Returns:
            If return_confidence=False: emotion label string
            If return_confidence=True: dict with prediction, confidence, and all probabilities
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")

        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Cannot predict.")

        self.model.eval()

        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        emotion = self.EMOTION_LABELS[predicted_class]

        if not return_confidence:
            return emotion

        # Return detailed prediction with all confidence scores
        all_confidences = {
            self.EMOTION_LABELS[i]: probabilities[0, i].item()
            for i in range(len(self.EMOTION_LABELS))
        }

        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_probabilities": all_confidences,
        }

    def predict_batch(
        self, texts: List[str], return_confidence: bool = False
    ) -> List[Union[str, Dict]]:
        """
        Efficient batch prediction for multiple texts.
        Essential for processing chat history or multiple user inputs.

        Args:
            texts: List of input text strings
            return_confidence: Whether to include confidence scores

        Returns:
            List of predictions (format depends on return_confidence)
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings")

        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Cannot predict.")

        self.model.eval()
        results = []

        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)

                for j, pred_class in enumerate(predicted_classes):
                    emotion = self.EMOTION_LABELS[pred_class.item()]
                    conf = probabilities[j, pred_class].item()

                    if return_confidence:
                        results.append({"emotion": emotion, "confidence": conf})
                    else:
                        results.append(emotion)

        return results

    def save_model(self, output_dir: str, metadata: Optional[Dict] = None) -> None:
        """
        Save the fine-tuned model with metadata for versioning and compliance.

        Args:
            output_dir: Directory path to save model artifacts
            metadata: Optional metadata (training config, performance metrics, etc.)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized. Cannot save.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving model to: {output_path}")

        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save metadata for tracking
        metadata_dict = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "emotion_labels": self.EMOTION_LABELS,
            "training_history": self.training_history,
            "save_timestamp": datetime.now().isoformat(),
            "device": str(self.device),
        }

        if metadata:
            metadata_dict.update(metadata)

        # Save metadata as JSON
        import json

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        logging.info(f"Model and metadata saved successfully to {output_path}")

    def load_model(self, model_dir: str) -> None:
        """
        Load a pre-trained model from disk.

        Args:
            model_dir: Directory containing saved model artifacts
        """
        model_path = Path(model_dir)

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        logging.info(f"Loading model from: {model_path}")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Cannot load model from {model_path}: {e}") from e


def main():
    """
    Example training pipeline demonstrating complete workflow.
    Adjust hyperparameters based on your computational resources.
    """
    logging.info("Starting Emotion Classifier Training Pipeline")

    # Initialize classifier
    classifier = EmotionClassifier(
        model_name="distilbert-base-uncased", max_length=128
    )

    # Prepare data
    train_loader, val_loader, test_loader = classifier.prepare_data(
        dataset_name="dair-ai/emotion", batch_size=32
    )

    # Train model
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=2e-5,
        early_stopping_patience=3,
        use_class_weights=True,
    )

    # Evaluate on test set
    test_metrics = classifier.evaluate(test_loader, show_confusion_matrix=True)

    # Save trained model
    classifier.save_model(
        output_dir="./models/emotion_classifier",
        metadata={"test_f1": test_metrics["f1_weighted"]},
    )

    # Demo predictions
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    test_texts = [
        "I'm so happy today! Everything is going great!",
        "I'm really worried about the exam tomorrow",
        "This makes me so angry and frustrated",
        "I miss you so much, thinking of you always",
    ]

    for text in test_texts:
        result = classifier.predict(text, return_confidence=True)
        print(f"\nText: {text}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")


if __name__ == "__main__":
    main()
