import json
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from config import config


def load_index():
    return json.loads(config.index_file.read_text(encoding="utf-8"))


def build_index(dry_run=False):
    files = list_knowledge_files(config.knowledge_dir)

    if not files:
        raise RuntimeError(f"No knowledge files found under {config.knowledge_dir}")

    raw_documents = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        relative_path = file_path.relative_to(config.content_root).as_posix()
        chunks = chunk_document(
            text,
            {
                "filePath": relative_path,
                "title": title_from_file(relative_path),
            },
        )
        raw_documents.extend(chunks)

    if not raw_documents:
        raise RuntimeError("Knowledge files were found, but no chunks were produced.")

    if dry_run:
        return {
            "fileCount": len(files),
            "chunkCount": len(raw_documents),
            "documents": raw_documents,
        }

    embeddings = embed_texts([doc["text"] for doc in raw_documents])
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    output = {
        "createdAt": created_at,
        "model": config.embed_model,
        "baseUrl": config.ollama_base_url,
        "chunkCount": len(raw_documents),
        "documents": [
            {
                **doc,
                "embedding": embeddings[index],
            }
            for index, doc in enumerate(raw_documents)
        ],
    }

    config.index_dir.mkdir(parents=True, exist_ok=True)
    config.index_file.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return output


def rank_documents(documents, query_embedding, top_k=None):
    limit = config.top_k if top_k is None else top_k
    ranked = []

    for doc in documents:
        ranked.append(
            {
                **doc,
                "score": dot_product(doc["embedding"], query_embedding),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def ensure_index_exists():
    try:
        return load_index()
    except FileNotFoundError:
        return build_index()


def list_knowledge_files(directory):
    files = []

    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            files.append(path)

    return files


def chunk_document(text, metadata):
    normalized = normalize_text(text)

    if not normalized:
        return []

    paragraphs = [item.strip() for item in normalized.split("\n\n") if item.strip()]
    chunks = []
    buffer = ""
    order = 0
    target_size = 900

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph

        if len(candidate) <= target_size:
            buffer = candidate
            continue

        if buffer:
            order += 1
            chunks.append(build_chunk(buffer, metadata, order))
            buffer = paragraph
            continue

        sentence_parts = [item.strip() for item in split_sentences(paragraph) if item.strip()]
        sentence_buffer = ""

        for sentence in sentence_parts:
            sentence_candidate = f"{sentence_buffer} {sentence}" if sentence_buffer else sentence

            if len(sentence_candidate) <= target_size:
                sentence_buffer = sentence_candidate
                continue

            if sentence_buffer:
                order += 1
                chunks.append(build_chunk(sentence_buffer, metadata, order))

            sentence_buffer = sentence

        if sentence_buffer:
            order += 1
            chunks.append(build_chunk(sentence_buffer, metadata, order))

        buffer = ""

    if buffer:
        order += 1
        chunks.append(build_chunk(buffer, metadata, order))

    return chunks


def build_chunk(text, metadata, order):
    return {
        "id": f"{metadata['filePath']}#{order}",
        "filePath": metadata["filePath"],
        "title": metadata["title"],
        "order": order,
        "text": text,
    }


def normalize_text(text):
    return text.replace("\r\n", "\n").replace("\ufeff", "").strip()


def split_sentences(paragraph):
    sentence = []
    parts = []

    for char in paragraph:
        sentence.append(char)
        if char in ".!?":
            parts.append("".join(sentence).strip())
            sentence = []

    remainder = "".join(sentence).strip()
    if remainder:
        parts.append(remainder)

    return parts


def title_from_file(file_path):
    return Path(file_path).stem


def dot_product(left, right):
    total = 0.0

    for left_value, right_value in zip(left, right):
        total += left_value * right_value

    return total


def embed_texts(input_texts):
    payload = json.dumps(
        {
            "model": config.embed_model,
            "input": input_texts,
        }
    ).encode("utf-8")

    req = request.Request(
        f"{config.ollama_base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.ollama_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embed request failed ({exc.code}): {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama embed request failed: {exc.reason}") from exc

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(input_texts):
        raise RuntimeError("Ollama returned an unexpected embeddings payload.")

    return embeddings
