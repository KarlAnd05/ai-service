import os
from dataclasses import dataclass
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTENT_ROOT = SERVICE_ROOT / "ai"
content_root_env = os.environ.get("AI_CONTENT_ROOT")
CONTENT_ROOT = (SERVICE_ROOT / content_root_env).resolve() if content_root_env else DEFAULT_CONTENT_ROOT.resolve()


@dataclass(frozen=True)
class Config:
    service_root: Path = SERVICE_ROOT
    content_root: Path = CONTENT_ROOT
    knowledge_dir: Path = CONTENT_ROOT / "knowledge"
    index_dir: Path = CONTENT_ROOT / "index"
    index_file: Path = CONTENT_ROOT / "index" / "rag-index.json"
    port: int = int(os.environ.get("AI_SERVICE_PORT", "4300"))
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    chat_model: str = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")
    embed_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma")
    top_k: int = int(os.environ.get("RAG_TOP_K", "4"))
    ollama_timeout_seconds: int = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "35"))


config = Config()
