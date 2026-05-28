import json
import math
import re
import unicodedata
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "ai" / "training" / "intent-model.json"
INTENT_REPLY_CONFIG = {
    "greeting": {
        "min_confidence": 0.35,
        "reply": "Hello. I'm here with you. Tell me what's happening so I can help you.",
    },
    "letting_go": {
        "min_confidence": 0.60,
        "reply": (
            "That sounds like a strong decision. Letting go can be a powerful step toward protecting your peace "
            "and making space for a new chapter. Tell me what's happening so I can support you."
        ),
    },
    "cheating_betrayal": {
        "min_confidence": 0.60,
        "reply": (
            "I'm really sorry you're dealing with this. Being cheated on can feel shocking, embarrassing, angry, "
            "and confusing, sometimes all at once. Don't pressure yourself to act like it doesn't hurt. "
            "It's okay if today feels bad. How can I help you? You can tell me what you want."
        ),
    },
    "betrayal_trust": {
        "min_confidence": 0.60,
        "reply": (
            "I'm really sorry you're going through that. When trust is broken, it can leave you feeling hurt, "
            "confused, and unsettled all at once. You do not have to force yourself to know what to do right away. "
            "If you want, you can tell me what happened and what feels hardest right now."
        ),
    },
    "relationship_distress": {
        "min_confidence": 0.60,
        "reply": (
            "I'm sorry you're going through this. Relationship pain can feel heavy, but you do not have to "
            "carry it alone. Tell me what's happening so I can support you."
        ),
    },
    "anxiety_support": {
        "min_confidence": 0.60,
        "reply": (
            "That sounds overwhelming. We can slow this down together and focus on one small step at a time. "
            "Tell me what's happening so I can help you feel a little steadier."
        ),
    },
    "panic_after_betrayal": {
        "min_confidence": 0.55,
        "reply": (
            "I'm sorry — that sounds really intense, especially on top of finding out about the cheating.\n\n"
            "Right now, focus on getting your body to feel safer:\n\n"
            "Put both feet on the floor.\n"
            "Look around and name 5 things you can see.\n"
            "Take slow breaths — don't force deep breaths, just make the exhale a little longer than the inhale.\n"
            "Sip some water if you can.\n"
            "If you're alone and don't want to be, text or call someone you trust.\n\n"
            "A panic attack can feel terrifying, but it usually peaks and then comes down."
        ),
    },
    "panic_attack": {
        "min_confidence": 0.60,
        "reply": (
            "I'm sorry — that sounds really intense.\n\n"
            "Right now, focus on getting your body to feel safer:\n\n"
            "Put both feet on the floor.\n"
            "Look around and name 5 things you can see.\n"
            "Take slow breaths — don't force deep breaths, just make the exhale a little longer than the inhale.\n"
            "Sip some water if you can.\n"
            "If you're alone and don't want to be, text or call someone you trust.\n\n"
            "A panic attack can feel terrifying, but it usually peaks and then comes down."
        ),
    },
    "presentation_anxiety": {
        "min_confidence": 0.60,
        "reply": (
            "It's completely normal to feel stressed before a presentation. That does not mean you are not "
            "capable. If you prepare well, take a deep breath, and focus on one part at a time, you will likely "
            "do better than you think. You've got this."
        ),
    },
    "sadness_low_mood": {
        "min_confidence": 0.60,
        "reply": (
            "I'm really sorry you're feeling this heavy right now. You do not have to push your feelings away or "
            "handle everything at once. We can take this one small step at a time. If you want, tell me what "
            "today has felt like for you."
        ),
    },
    "loneliness": {
        "min_confidence": 0.60,
        "reply": (
            "That sounds really lonely, and I'm sorry you're carrying that feeling. Wanting connection does not "
            "make you weak. I'm here with you, and you can tell me what has been feeling most isolating."
        ),
    },
    "overwhelmed_general": {
        "min_confidence": 0.60,
        "reply": (
            "That sounds like a lot to carry all at once. When everything feels heavy, we do not need to solve it "
            "all right now. We can slow it down together. Tell me what feels like the hardest part at this moment."
        ),
    },
}
EXAM_SUPPORT_REPLY = (
    "Ah, exams and feeling tired is a very different situation.\n\n"
    "If you're tired because you've been studying, don't force yourself to push through completely exhausted. "
    "After a point your brain stops retaining much.\n\n"
    "Try this:\n\n"
    "If you're sleepy -> sleep 60-90 min, or go to bed if it's late. Sleep usually helps memory more than "
    "another exhausted study session.\n"
    "If you're mentally drained but awake -> take a 15-20 min break, water and snack, then do one small study "
    "block of 25-40 min.\n"
    "Focus on:\n"
    "topics most likely to appear\n"
    "exercises or past questions\n"
    "summaries instead of rereading everything"
)

_cached_model = None


def tokenize(text):
    normalized = unicodedata.normalize("NFD", str(text or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9']+", normalized.lower())
    bigrams = [f"{words[index]}_{words[index + 1]}" for index in range(len(words) - 1)]
    return [*words, *bigrams]


def load_model():
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    if not MODEL_PATH.exists():
        return None

    parsed = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    _cached_model = parsed if isinstance(parsed, dict) else None
    return _cached_model


def classify_intent(message):
    model = load_model()

    if not model:
        return None

    tokens = tokenize(message)
    if not tokens:
        return None

    vocabulary_size = max(int(model.get("vocabularySize", 0) or 0), 1)
    labels = model.get("labels", [])
    scores = []

    for label in labels:
        doc_count = int(model.get("classDocCounts", {}).get(label, 0) or 0)
        total_docs = max(int(model.get("totalDocs", 0) or 0), 1)
        prior = math.log((doc_count / total_docs) if doc_count else (1 / total_docs))
        token_total = int(model.get("classTokenTotals", {}).get(label, 0) or 0)
        token_counts = model.get("tokenCounts", {}).get(label, {})

        score = prior
        for token in tokens:
            token_count = int(token_counts.get(token, 0) or 0)
            score += math.log((token_count + 1) / (token_total + vocabulary_size))

        scores.append({"label": label, "score": score})

    max_score = max(entry["score"] for entry in scores)
    normalized_scores = []

    for entry in scores:
        probability = math.exp(entry["score"] - max_score)
        normalized_scores.append({**entry, "probability": probability})

    probability_sum = sum(entry["probability"] for entry in normalized_scores) or 1
    ranked = sorted(
        [
            {
                "label": entry["label"],
                "confidence": entry["probability"] / probability_sum,
                "score": entry["score"],
            }
            for entry in normalized_scores
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )

    top = ranked[0] if ranked else {"label": None, "confidence": 0}
    return {
        "label": top["label"],
        "confidence": top["confidence"],
        "ranked": ranked,
    }


def build_intent_reply(prediction):
    if not prediction:
        return None

    label = prediction.get("label")
    confidence = prediction.get("confidence", 0)

    if label == "exam_stress" and confidence >= 0.22:
        return EXAM_SUPPORT_REPLY

    config = INTENT_REPLY_CONFIG.get(label)
    if config and confidence >= config["min_confidence"]:
        return config["reply"]

    return None
