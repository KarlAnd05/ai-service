# AI Service

This is a standalone AI project for AskDrScott.

It is separate from the web and mobile frontends and is meant to be used by both.
The AI backend source now runs in Python, while the knowledge and training data stay in Markdown and JSON.

## Folder structure

```text
ai-service/
  ai/
    knowledge/
    index/
  src/
  docs/
  package.json
```

## What it does

- stores the AI knowledge files locally
- rebuilds the RAG index
- connects to Ollama
- exposes chat endpoints for both web and mobile

## Run it

```powershell
python src/train_intent_model.py
python src/rebuild_index.py
python src/server.py
```

You can also keep using the npm shortcuts:

```powershell
npm run train:intents
npm run index
npm run dev
```

## Default URL

```text
http://localhost:4300
```

## Main endpoint

```text
POST /api/chat/reply
```
