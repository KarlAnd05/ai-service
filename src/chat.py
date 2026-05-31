import json
import socket
from urllib import error, request

from config import config
from crisis import build_crisis_reply, is_crisis_message


def embed_query(question):
    data = post_json(
        f"{config.ollama_base_url}/api/embed",
        {
            "model": config.embed_model,
            "input": question,
        },
        "embed",
    )

    embeddings = data.get("embeddings")
    embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None

    if not isinstance(embedding, list):
        raise RuntimeError("Ollama returned an unexpected embedding for the question.")

    return embedding


def chat_with_context(coach_name, question, context_text, history=None):
    history = history or []
    system = " ".join(
        [
            f"You are {coach_name}.",
            "You are a supportive wellness chatbot using approved project documents.",
            "Use the provided context when answering.",
            "If the context does not support a claim, say you are not sure instead of guessing.",
            "Do not diagnose and do not give medication advice.",
            "If the user appears in crisis, prioritize safety and encourage immediate human help.",
            "Keep the answer supportive, calm, and practical.",
            "If the user message is only a simple greeting such as hello, hi, or hey, reply in one short sentence when possible and no more than two short sentences.",
        ]
    )

    history_lines = []
    for turn in history:
        role = "Assistant" if turn.get("role") == "assistant" else "User"
        content = str(turn.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_lines = history_lines[-6:]
    trimmed_context = (context_text or "").strip()
    if len(trimmed_context) > 1400:
        trimmed_context = trimmed_context[:1400].rstrip() + "\n...[truncated]"

    prompt_parts = []

    if history_lines:
        prompt_parts.append("Recent conversation:\n" + "\n".join(history_lines))

    prompt_parts.append(f"Approved context:\n{trimmed_context or 'No approved context found.'}")
    prompt_parts.append(f"User question:\n{question}")
    prompt_parts.append(
        "Answer in a short, supportive way. Base the answer on the approved context when it is helpful. "
        "If the context does not support a claim, say you are not sure instead of guessing."
    )

    prompt = "\n\n".join(prompt_parts)

    data = post_json(
        f"{config.ollama_base_url}/api/generate",
        {
            "model": config.chat_model,
            "stream": False,
            "system": system,
            "prompt": prompt,
            "options": {
                "num_predict": 180,
                "temperature": 0.4,
            },
        },
        "generate",
    )
    content = data.get("response")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an unexpected generate response.")

    return content.strip()


def build_fallback_reply(coach_name, user_message):
    safe_coach_name = coach_name.strip() if isinstance(coach_name, str) and coach_name.strip() else "Coach"
    normalized_coach_name = safe_coach_name.lower()
    normalized_message = (user_message or "").strip().lower()

    def mentions_any(*terms):
        return any(term in normalized_message for term in terms)

    if is_crisis_message(user_message):
        return build_crisis_reply()

    if not normalized_message or mentions_any("hello", "hi", "hey"):
        return "Hello. I'm here with you. Tell me what's happening so I can help you."

    if mentions_any("forget", "move on", "let go", "new life"):
        return (
            "That sounds like a strong decision. Letting go can be a powerful step toward protecting "
            "your peace and making space for a new chapter. Tell me what's happening so I can support you."
        )

    if "relationship" in normalized_coach_name or mentions_any(
        "boyfriend",
        "girlfriend",
        "relationship",
        "break up",
        "breakup",
        "ex",
    ):
        return (
            "I'm sorry you're going through this. Relationship pain can feel heavy, but you do not have "
            "to carry it alone. Tell me what's happening so I can support you."
        )

    if "anxiety" in normalized_coach_name or mentions_any(
        "anxious",
        "anxiety",
        "panic",
        "stress",
        "exam",
        "overthinking",
        "chest feels tight",
    ):
        return (
            "That sounds overwhelming. We can slow this down together and focus on one small step at a time. "
            "Tell me what's happening so I can help you feel a little steadier."
        )

    return f"I'm here with you. Tell me what's happening so {safe_coach_name} can support you."


def post_json(url, payload, action_name):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.ollama_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama {action_name} request failed ({exc.code}): {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama {action_name} request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama {action_name} request timed out after {config.ollama_timeout_seconds} seconds."
        ) from exc
    except socket.timeout as exc:
        raise RuntimeError(
            f"Ollama {action_name} request timed out after {config.ollama_timeout_seconds} seconds."
        ) from exc
