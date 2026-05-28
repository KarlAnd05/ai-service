# Ollama RAG Starter Pack

This folder contains the starter content and generated index for the chatbot's RAG setup.

## Folders

- `knowledge/`
  Approved content that the chatbot is allowed to use.
- `examples/`
  Example test prompts and sample interactions for evaluation.
- `index/`
  Generated retrieval index created from the knowledge files.

## Workflow

1. Edit the files in `knowledge/` so they match your doctor's approved content.
2. Run `npm run ai:index` to build the local RAG index with Ollama embeddings.
3. Run `npm run ai:test -- "your question" "Coach Name"` to test retrieval and answer quality.
4. Rebuild the index any time the knowledge files change.
