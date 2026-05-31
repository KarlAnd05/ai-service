import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from chat import build_fallback_reply, chat_with_context, embed_query
from config import config
from crisis import build_crisis_reply, is_crisis_message
from intent_classifier import build_intent_reply, classify_intent
from rag import build_index, ensure_index_exists, rank_documents


JSON_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}

SIMPLE_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "hey there",
}


def build_greeting_reply(display_name=""):
    safe_name = display_name.strip() if isinstance(display_name, str) else ""
    if safe_name:
        return f"Hello, {safe_name}. I'm here with you. Tell me what's happening so I can help you."

    return "Hello. I'm here with you. Tell me what's happening so I can help you."


class AIServiceHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_empty(204)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "askdrscott-ai-service",
                    "ollamaBaseUrl": config.ollama_base_url,
                    "contentRoot": str(config.content_root),
                },
            )
            return

        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/index/rebuild":
                result = build_index()
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "chunkCount": result["chunkCount"],
                        "model": result["model"],
                    },
                )
                return

            if parsed.path == "/api/chat/reply":
                body = self.read_json_body()
                coach_name = (
                    body.get("coachName", "").strip()
                    if isinstance(body.get("coachName"), str) and body.get("coachName", "").strip()
                    else "General Mental Health Coach"
                )
                display_name = body.get("displayName", "").strip() if isinstance(body.get("displayName"), str) else ""
                message = body.get("message", "").strip() if isinstance(body.get("message"), str) else ""
                history = body.get("history", []) if isinstance(body.get("history"), list) else []
                history = history[-8:]

                if not message:
                    self.send_json(400, {"error": "message is required"})
                    return

                if message.lower() in SIMPLE_GREETINGS:
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "coachName": coach_name,
                            "reply": build_greeting_reply(display_name),
                            "usedFallback": False,
                            "usedIntentModel": True,
                            "intent": {
                                "label": "greeting",
                                "confidence": 1,
                                "ranked": [],
                            },
                            "sources": [],
                        },
                    )
                    return

                if is_crisis_message(message):
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "coachName": coach_name,
                            "reply": build_crisis_reply(),
                            "usedFallback": False,
                            "usedIntentModel": False,
                            "intent": {
                                "label": "crisis",
                                "confidence": 1,
                                "ranked": [],
                            },
                            "sources": [
                                {
                                    "id": "knowledge/shared/crisis-escalation.md",
                                    "filePath": "knowledge/shared/crisis-escalation.md",
                                    "title": "crisis-escalation",
                                    "score": 1,
                                }
                            ],
                        },
                    )
                    return

                intent_prediction = classify_intent(message)
                trained_reply = build_intent_reply(intent_prediction, message)

                if trained_reply:
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "coachName": coach_name,
                            "reply": trained_reply,
                            "usedFallback": False,
                            "usedIntentModel": True,
                            "intent": intent_prediction,
                            "sources": [],
                        },
                    )
                    return

                used_fallback = False
                matches = []

                try:
                    index = ensure_index_exists()
                    query_embedding = embed_query(message)
                    matches = rank_documents(index["documents"], query_embedding, config.top_k)
                    context_text = "\n\n".join(
                        [
                            f"[Source {idx + 1}] {match['filePath']}\n{match['text']}"
                            for idx, match in enumerate(matches)
                        ]
                    )

                    reply = chat_with_context(
                        coach_name=coach_name,
                        question=message,
                        context_text=context_text,
                        history=history,
                    )
                except Exception as error:
                    used_fallback = True
                    reply = build_fallback_reply(coach_name, message)
                    print(error)

                self.send_json(
                    200,
                    {
                        "ok": True,
                        "coachName": coach_name,
                        "reply": reply,
                        "usedFallback": used_fallback,
                        "usedIntentModel": False,
                        "intent": intent_prediction,
                        "sources": [
                            {
                                "id": match["id"],
                                "filePath": match["filePath"],
                                "title": match["title"],
                                "score": round(match["score"], 4),
                            }
                            for match in matches
                        ],
                    },
                )
                return

            self.send_json(404, {"error": "Not found"})
        except Exception as error:
            print(error)
            self.send_json(500, {"error": "Internal server error"})

    def send_json(self, status_code, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        for header, value in JSON_HEADERS.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)

    def send_empty(self, status_code):
        self.send_response(status_code)
        for header, value in JSON_HEADERS.items():
            self.send_header(header, value)
        self.end_headers()

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}

        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer(("0.0.0.0", config.port), AIServiceHandler)
    print(f"AI service running on http://localhost:{config.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
