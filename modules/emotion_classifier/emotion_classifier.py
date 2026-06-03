"""
Emotion Classification Module
Fine-tuned DistilBERT for 6-class emotion detection in mental health chatbot contexts.

Author:
Date: May 2026
"""

import json
import logging
import warnings
from copy import deepcopy
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

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """Wraps tokenized text and integer labels for use with PyTorch DataLoader."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class EmotionClassifier:
    """
    Fine-tunes DistilBERT for 6-class emotion detection.

    Covers the full lifecycle: data loading, training with early stopping,
    evaluation, single and batch inference, and model persistence.

    Emotion labels follow the dair-ai/emotion dataset convention:
        0: sadness  1: joy  2: love  3: anger  4: fear  5: surprise
    """

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
        Args:
            model_name: Any HuggingFace sequence-classification compatible model.
            max_length: Token sequence length. 128 covers ~95% of conversational text.
            device: 'cuda', 'cpu', or None to auto-detect.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }

        logger.info(f"Loading tokenizer '{model_name}' on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load tokenizer for '{model_name}'") from exc

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        dataset_name: str = "dair-ai/emotion",
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the dair-ai/emotion dataset and returns DataLoaders for each split.

        The dataset ships with its own train/validation/test splits, so no
        manual splitting is needed here.

        Args:
            dataset_name: HuggingFace dataset identifier.
            batch_size: Samples per batch. Reduce to 16 if you hit CUDA OOM.
            num_workers: Worker processes for data loading. Keep at 0 inside
                         Docker containers or Jupyter notebooks to avoid forking
                         issues; bump to 2-4 on a bare-metal GPU machine.

        Returns:
            (train_loader, val_loader, test_loader)
        """
        logger.info(f"Loading dataset '{dataset_name}'")
        try:
            dataset = load_dataset(dataset_name)
        except Exception as exc:
            raise ValueError(f"Could not load dataset '{dataset_name}'") from exc

        splits = {
            name: EmotionDataset(
                texts=dataset[name]["text"],
                labels=dataset[name]["label"],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            for name in ("train", "validation", "test")
        }

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        )

        loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
        train_loader = DataLoader(splits["train"], shuffle=True, **loader_kwargs)
        val_loader = DataLoader(splits["validation"], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(splits["test"], shuffle=False, **loader_kwargs)

        return train_loader, val_loader, test_loader

    def _class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Computes inverse-frequency class weights from the training set.

        Emotion datasets tend to be skewed toward sadness and joy, which can
        cause the model to under-learn minority classes like surprise or love.
        Weighted loss counteracts this without requiring oversampling.
        """
        counts = torch.zeros(len(self.EMOTION_LABELS))
        for batch in train_loader:
            for label in batch["label"]:
                counts[label] += 1

        weights = counts.sum() / (len(self.EMOTION_LABELS) * counts)
        logger.info("Class weights: %s", weights.tolist())
        return weights.to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
        Fine-tunes the model with AdamW, linear LR warmup, gradient clipping,
        and early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum number of passes over the training data.
            learning_rate: Peak LR for AdamW. 2e-5 is a safe default for
                           DistilBERT fine-tuning; go lower (1e-5) if loss
                           is unstable early on.
            warmup_steps: Steps over which LR linearly ramps up. Helps avoid
                          destroying pre-trained weights in the first batches.
            early_stopping_patience: Stop after this many epochs with no
                                     improvement in validation F1.
            use_class_weights: Pass weighted loss to the criterion. Recommended
                               unless your dataset is already balanced.

        Returns:
            Training history with per-epoch train loss, val loss, and val F1.
        """
        logger.info("Initializing model '%s'", self.model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(self.EMOTION_LABELS)
            ).to(self.device)
        except Exception as exc:
            raise RuntimeError(f"Could not load model '{self.model_name}'") from exc

        criterion = nn.CrossEntropyLoss(
            weight=self._class_weights(train_loader) if use_class_weights else None
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_loader) * epochs,
        )

        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = epoch_loss / len(train_loader)
            val_loss, val_f1, _ = self._evaluate(val_loader, criterion)

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_f1"].append(val_f1)

            logger.info(
                "Epoch %d/%d — train loss: %.4f  val loss: %.4f  val F1: %.4f",
                epoch, epochs, avg_train_loss, val_loss, val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = deepcopy(self.model.state_dict())
                patience_counter = 0
                logger.info("New best val F1: %.4f — checkpoint saved", best_val_f1)
            else:
                patience_counter += 1
                logger.info(
                    "No improvement (%d/%d)", patience_counter, early_stopping_patience
                )
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info("Restored best checkpoint (val F1: %.4f)", best_val_f1)

        return self.training_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, Dict]:
        """
        Runs a full pass over `loader` and returns loss, weighted F1, and a
        metrics dict. Used internally by both train() and evaluate().
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")

        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                total_loss += criterion(logits, labels).item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_weighted": weighted_f1,
            "predictions": all_preds,
            "labels": all_labels,
        }
        return avg_loss, weighted_f1, metrics

    def evaluate(
        self,
        test_loader: DataLoader,
        show_confusion_matrix: bool = True,
    ) -> Dict:
        """
        Prints a full classification report and optionally a confusion matrix.

        Uses unweighted CrossEntropyLoss here so the reported loss reflects
        raw model performance rather than the training objective.

        Args:
            test_loader: Test DataLoader.
            show_confusion_matrix: Print the confusion matrix after the report.

        Returns:
            Metrics dict (loss, accuracy, f1_weighted, predictions, labels).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")

        logger.info("Evaluating on test set...")
        _, _, metrics = self._evaluate(test_loader, nn.CrossEntropyLoss())

        divider = "=" * 70
        print(f"\n{divider}")
        print(f"{'TEST SET RESULTS':^70}")
        print(divider)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print("\nPer-class breakdown:")
        print("-" * 70)
        print(
            classification_report(
                metrics["labels"],
                metrics["predictions"],
                target_names=list(self.EMOTION_LABELS.values()),
                digits=4,
            )
        )

        if show_confusion_matrix:
            cm = confusion_matrix(metrics["labels"], metrics["predictions"])
            print("Confusion matrix (rows = true, columns = predicted):")
            print("-" * 70)
            print(cm)

        print(f"{divider}\n")
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        return_confidence: bool = True,
    ) -> Union[str, Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predicts the emotion of a single text string.

        Args:
            text: Raw input text.
            return_confidence: If False, returns just the label string.
                               If True, returns a dict with the predicted emotion,
                               its confidence score, and probabilities for all classes.

        Returns:
            str or dict depending on return_confidence.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")

        self.model.eval()
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        emotion = self.EMOTION_LABELS[pred_idx]

        if not return_confidence:
            return emotion

        return {
            "emotion": emotion,
            "confidence": probs[0, pred_idx].item(),
            "all_probabilities": {
                self.EMOTION_LABELS[i]: probs[0, i].item()
                for i in range(len(self.EMOTION_LABELS))
            },
        }

    def predict_batch(
        self,
        texts: List[str],
        return_confidence: bool = False,
        batch_size: int = 32,
    ) -> List[Union[str, Dict]]:
        """
        Runs inference over a list of strings in mini-batches.

        Useful for scoring a chat history or a queue of incoming messages
        without the overhead of one tokenization call per text.

        Args:
            texts: List of raw input strings.
            return_confidence: Include confidence score per prediction.
            batch_size: Internal mini-batch size. Reduce if memory is tight.

        Returns:
            List of emotion strings, or dicts if return_confidence=True.
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings.")
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")

        self.model.eval()
        results = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encodings = self.tokenizer(
                chunk,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                probs = torch.softmax(logits, dim=1)
                pred_indices = torch.argmax(probs, dim=1)

            for j, idx in enumerate(pred_indices):
                emotion = self.EMOTION_LABELS[idx.item()]
                if return_confidence:
                    results.append({"emotion": emotion, "confidence": probs[j, idx].item()})
                else:
                    results.append(emotion)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, output_dir: str, metadata: Optional[Dict] = None) -> None:
        """
        Saves model weights, tokenizer, and a metadata JSON to output_dir.

        The metadata file records the base model name, max_length, label mapping,
        training history, and a save timestamp — enough to reproduce or audit
        the checkpoint later.

        Args:
            output_dir: Destination directory (created if it does not exist).
            metadata: Any additional key-value pairs to merge into metadata.json.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Nothing to save. Train or load a model first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model to '%s'", output_path)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        meta = {
            "base_model": self.model_name,
            "max_length": self.max_length,
            "emotion_labels": self.EMOTION_LABELS,
            "training_history": self.training_history,
            "saved_at": datetime.now().isoformat(),
            "device": str(self.device),
        }
        if metadata:
            meta.update(metadata)

        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved successfully.")

    def load_model(self, model_dir: str) -> None:
        """
        Loads a saved checkpoint from disk.

        Args:
            model_dir: Directory produced by save_model().
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        logger.info("Loading model from '%s'", model_path)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from '{model_path}'") from exc

        logger.info("Model loaded.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    classifier = EmotionClassifier(model_name="distilbert-base-uncased", max_length=128)

    train_loader, val_loader, test_loader = classifier.prepare_data(
        dataset_name="dair-ai/emotion",
        batch_size=32,
    )

    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=2e-5,
        early_stopping_patience=3,
        use_class_weights=True,
    )

    test_metrics = classifier.evaluate(test_loader, show_confusion_matrix=True)

    classifier.save_model(
        output_dir="./models/emotion_classifier",
        metadata={"test_f1": test_metrics["f1_weighted"]},
    )

    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    samples = [
        "I'm so happy today! Everything is going great!",
        "I'm really worried about the exam tomorrow.",
        "This makes me so angry and frustrated.",
        "I miss you so much, thinking of you always.",
    ]

    for text in samples:
        result = classifier.predict(text, return_confidence=True)
        print(f"\n  text:      {text}")
        print(f"  emotion:   {result['emotion']}  ({result['confidence']:.1%})")


if __name__ == "__main__":
    main()