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
  tools/terminal_linux.py shell tools: one-shot commands and persistent
                         pty sessions, gated by ACCESS_LEVEL
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

## Terminal access

The agent can run shell commands on the host — one-shot with `run_command`, or
in a persistent bash session on a pty (`terminal_open`, `terminal_run`,
`terminal_read`, `terminal_send_keys`, `terminal_interrupt`, `terminal_list`,
`terminal_close`) that keeps its working directory and environment between
calls. How much it may do is set by `ACCESS_LEVEL` in
`src/amadeus_amth/tools/terminal_linux.py`, or by `AMADEUS_TERMINAL_ACCESS` in
`.env`:

| Level | What it allows |
| --- | --- |
| `RESTRICTED` | An allowlist of read-only commands, run without a shell, inside the workspace only. No pipes, redirection or chaining. |
| `MEDIUM` (default) | A full shell anywhere on the filesystem, minus anything that administers the box: sudo, package installs, service/user/firewall/mount changes, containers, writes into system directories. |
| `UNRESTRICTED` | Everything except a short list of irreversible commands — wiping `/`, formatting a disk, powering off — which are refused at every level. |

The levels are pattern checks on a command line: a guard rail against a
careless model, not a sandbox and not a security boundary. Anything determined
to get around them can (a variable, base64, a script file, an interpreter). The
real boundary has to come from the OS — run Amadeus as an unprivileged user
with no sudo rights, in a container, with nothing mounted it has no business
reaching.
