# External AI Service Setup

This AI service now lives outside the frontend projects.

Suggested structure:

```text
AskDrScott/
  mobile-ui/
  web-ui/
  ai-service/
```

## Start Ollama

```powershell
ollama list
```

## Start the AI service

```powershell
cd "C:\Users\10User\Desktop\web net\AskDrScott\ai-service"
python src\train_intent_model.py
python src\rebuild_index.py
python src\server.py
```

Or with the existing shortcuts:

```powershell
npm run train:intents
npm run index
npm run dev
```

## Connect the web frontend

Set:

```text
VITE_AI_SERVICE_BASE_URL=http://localhost:4300
```

## Connect the mobile frontend

Use:

```text
POST http://<your-computer-ip>:4300/api/chat/reply
```

## Important

Keep future AI changes in this separate `ai-service` project so both frontends use the same logic.
