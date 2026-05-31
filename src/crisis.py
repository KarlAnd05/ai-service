import re
import unicodedata


CRISIS_PATTERNS = [
    r"\bi want to (die|kill myself|end my life|commit suicide|suicide)\b",
    r"\bi (?:wanna|need to|have to) (die|kill myself|end my life|commit suicide|suicide)\b",
    r"\bi(?:'m| am)? suicidal\b",
    r"\bi want to (hurt|harm) myself\b",
    r"\bi(?:'m| am) going to (kill myself|hurt myself|harm myself|end my life)\b",
    r"\bi want to jump\b.*\b(?:from|off)\b.*\b(roof|rooftop|building|bridge|balcony|balcon|window|cliff|ledge|height|top)\b",
    r"\bi(?:'m| am) going to jump\b.*\b(?:from|off)\b.*\b(roof|rooftop|building|bridge|balcony|balcon|window|cliff|ledge|height|top)\b",
    r"\bi (?:have|made) a plan\b",
    r"\bi already took\b",
    r"\bi overdosed\b",
    r"\bi don't feel safe\b",
    r"\bi do not feel safe\b",
    r"\bi don't want to be here anymore\b",
    r"\bi do not want to be here anymore\b",
    r"\bi don't want to live\b",
    r"\bi do not want to live\b",
    r"\bi can't go on\b",
    r"\bi cannot go on\b",
    r"\bi want to (hurt|kill) someone\b",
]


def normalize_message(message):
    normalized = unicodedata.normalize("NFD", str(message or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.lower().strip()


def is_crisis_message(message):
    normalized = normalize_message(message)
    if not normalized:
        return False

    return any(re.search(pattern, normalized) for pattern in CRISIS_PATTERNS)


def build_crisis_reply():
    return (
        "I'm really sorry you're going through this. Your safety matters most right now. "
        "If you're in Lebanon, please call 1564 now. If you're in immediate danger or you do not "
        "feel you can stay safe, call 140 for ambulance support or 112/999 for police right now. "
        "If you're outside Lebanon, contact your local emergency services or a local crisis line "
        "immediately. If you can, reach out to a trusted person who can stay with you right now, "
        "and move away from anything you could use to hurt yourself."
    )
