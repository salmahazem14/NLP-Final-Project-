import json
from pathlib import Path

FEEDBACK_FILE = Path("data/feedback.json")


def load_feedback_counts():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "positive": 0,
            "negative": 0
        }


def save_feedback_counts(data):
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def record_feedback(vote: str):
    counts = load_feedback_counts()

    vote = vote.lower().strip()

    if vote == "up":
        counts["positive"] += 1

    elif vote == "down":
        counts["negative"] += 1

    else:
        raise ValueError(f"Invalid vote: {vote}")

    save_feedback_counts(counts)

    return {
        "status": "success",
        "positive": counts["positive"],
        "negative": counts["negative"]
    }