# Amadeus-AMTH

The production version of AMADEUS — an AI assistant with fully autonomous
capabilities, built to run in an isolated Docker Compose stack on a mini-server.

## Layout

```
src/amadeus_amth/
  __init__.py            main() — the uvicorn launcher behind the console script
  __main__.py            python -m amadeus_amth
  app.py                 FastAPI app: /chat, / and the heartbeat lifespan
  agent.py               Amadeus — the model turn/tool-call loop
  models.py              request schemas
  memory/str_memory.py   short-term rolling conversation memory
  tools/workspace_tool.py file tools, confined to the workspace root
scripts/smoke_chat.py    one-shot POST against a running instance
workspace/               the agent's own scratch space (it writes here)
chat.html                local dev client (gitignored)
```

## Setup

Requires Python 3.14, [uv](https://docs.astral.sh/uv/), and a reachable
[Ollama](https://ollama.com) instance.

```sh
uv sync
cp .env.example .env    # then fill in BASE_MODEL at minimum
```

## Run

```sh
uv run amadeus-amth     # or: uv run python -m amadeus_amth
```

Serves on `http://127.0.0.2:8000`. `POST /chat` takes `{"message": "..."}` and
returns `{"response": "..."}`; `GET /` is a liveness check.

```sh
uv run python scripts/smoke_chat.py
```

Open `chat.html` in a browser for a local UI against the same endpoint.

## Configuration

All optional — see `.env.example` for the full list and defaults. `BASE_MODEL`
and `SYSTEM_PROMPT` control the agent; `AMADEUS_WORKSPACE` relocates the
directory the file tools are allowed to touch.
