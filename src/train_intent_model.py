import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent.parent / "ai" / "training" / "intent-dataset.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "ai" / "training" / "intent-model.json"


def tokenize(text):
    normalized = unicodedata.normalize("NFD", str(text or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9']+", normalized.lower())
    bigrams = [f"{words[index]}_{words[index + 1]}" for index in range(len(words) - 1)]
    return [*words, *bigrams]


def normalize_label(label):
    normalized = str(label or "").strip().lower()
    return normalized or "general_support"


def train_model(examples):
    labels = list(dict.fromkeys(example["label"] for example in examples))
    class_doc_counts = {label: 0 for label in labels}
    class_token_totals = {label: 0 for label in labels}
    token_counts = {label: {} for label in labels}
    vocabulary = set()

    for example in examples:
        label = example["label"]
        tokens = tokenize(example["text"])
        class_doc_counts[label] += 1

        for token in tokens:
            vocabulary.add(token)
            token_counts[label][token] = token_counts[label].get(token, 0) + 1
            class_token_totals[label] += 1

    return {
        "labels": labels,
        "totalDocs": len(examples),
        "vocabularySize": len(vocabulary),
        "classDocCounts": class_doc_counts,
        "classTokenTotals": class_token_totals,
        "tokenCounts": token_counts,
    }


def predict(model, text):
    tokens = tokenize(text)
    vocabulary_size = max(int(model.get("vocabularySize", 0) or 0), 1)
    best = None

    for label in model.get("labels", []):
        doc_count = int(model.get("classDocCounts", {}).get(label, 0) or 0)
        total_docs = max(int(model.get("totalDocs", 0) or 0), 1)
        prior = math.log((doc_count / total_docs) if doc_count else (1 / total_docs))
        token_total = int(model.get("classTokenTotals", {}).get(label, 0) or 0)
        token_bag = model.get("tokenCounts", {}).get(label, {})

        score = prior
        for token in tokens:
            token_count = int(token_bag.get(token, 0) or 0)
            score += math.log((token_count + 1) / (token_total + vocabulary_size))

        if best is None or score > best["score"]:
            best = {"label": label, "score": score}

    return best["label"] if best else None


def split_dataset(examples):
    grouped = {}

    for example in examples:
        grouped.setdefault(example["label"], []).append(example)

    train = []
    test = []

    for entries in grouped.values():
        for index, example in enumerate(entries):
            if index % 4 == 0:
                test.append(example)
            else:
                train.append(example)

    return train, test


def evaluate_model(model, examples):
    if not examples:
        return {"accuracy": 1, "total": 0, "correct": 0}

    correct = 0
    for example in examples:
        if predict(model, example["text"]) == example["label"]:
            correct += 1

    return {
        "accuracy": correct / len(examples),
        "total": len(examples),
        "correct": correct,
    }


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if not isinstance(dataset, list) or not dataset:
        raise RuntimeError("intent-dataset.json must contain at least one example.")

    normalized_dataset = [
        {
            "text": example["text"],
            "label": normalize_label(example["label"]),
        }
        for example in dataset
    ]
    train, test = split_dataset(normalized_dataset)
    evaluation_model = train_model(train)
    metrics = evaluate_model(evaluation_model, test)
    final_model = train_model(normalized_dataset)
    final_model["metadata"] = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datasetSize": len(normalized_dataset),
        "trainSize": len(train),
        "testSize": len(test),
        "accuracy": round(metrics["accuracy"], 4),
        "correct": metrics["correct"],
        "total": metrics["total"],
        "labels": final_model["labels"],
    }

    OUTPUT_PATH.write_text(json.dumps(final_model, indent=2), encoding="utf-8")

    print(f"Intent model saved to {OUTPUT_PATH}")
    print(f"Labels: {', '.join(final_model['labels'])}")
    print(
        "Holdout accuracy: "
        f"{metrics['correct']}/{metrics['total']} ({metrics['accuracy'] * 100:.1f}%)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        raise SystemExit(1) from error
