Amadeus-AMTH
The production version of AMADEUS — an AI assistant with fully autonomous
capabilities, built to run in an isolated Docker Compose stack on a mini-server.
Layout
```
src/amadeus_amth/
  __init__.py             main() — the uvicorn launcher behind the console script
  __main__.py             python -m amadeus_amth
  app.py                  FastAPI app: POST /chat, GET /health, the heartbeat
                          lifespan and the web client mount
  agent.py                Amadeus — the model turn/tool-call loop; stamps each
                          user message with a timestamp as it enters memory
  models.py               request schemas
  memory/str_memory.py    short-term rolling conversation memory
  memory/lt_memory.py     LongTermMemory — durable facts in a local ChromaDB
                          store, embedded with bge-m3 via Ollama
  memory/semantic_memory/ ChromaDB's persistence directory (generated at
                          runtime, not source)
  tools/workspace_tool.py file tools, confined to the workspace root
  tools/terminal_linux.py shell tools: one-shot commands and persistent pty
                          sessions, gated by ACCESS_LEVEL; every call echoes a
                          console log line
  tools/memory_tool.py    save_memory / retrieve_memory — the active half of
                          long-term memory
  web/                    the web chat client (index.html plus a PWA manifest
                          and icon), served from the app root
scripts/basic_chat.py     interactive REPL chat client against a running instance
workspace/                the agent's own scratch space (it writes here;
                          runtime content, not source)
```
Setup
Requires Python 3.14, uv, and a reachable
Ollama instance.
Two Ollama models are needed:
the chat model named in `BASE_MODEL` (below), and
`bge-m3`, the embedding model long-term memory embeds every fact with —
`ollama pull bge-m3`.
```sh
uv sync
```
Create a `.env` next to `pyproject.toml` (there is no checked-in example) and
fill in `BASE_MODEL` at minimum. `SYSTEM_PROMPT`, `AMADEUS_WORKSPACE` and
`AMADEUS_TERMINAL_ACCESS` are optional.
Run
```sh
uv run amadeus-amth     # or: uv run python -m amadeus_amth
```
Serves on `http://127.0.0.2:8000`:
`GET /` — the web chat client, mounted last so it does not swallow the API
routes
`GET /health` — liveness check
`POST /chat` — takes `{"message": "..."}` and returns `{"response": "..."}`
Open `http://127.0.0.2:8000/` in a browser for the web UI. To chat from a
terminal instead:
```sh
uv run python scripts/basic_chat.py
```
— an interactive loop: type a message, get a reply, Ctrl-C to exit.
Heartbeat
While the server runs, a heartbeat loop in `app.py` wakes every 4 minutes
(hardcoded for now) and hands the agent a housekeeping directive instead of a
user message — its cue to autonomously tidy the workspace, review earlier
conversations, and store anything worth keeping in long-term memory.
Memory
Two layers, one passive lookup and one active:
Short-term (`memory/str_memory.py`) — a rolling window of the current
conversation. Each user message is stamped with a timestamp as it enters.
Long-term (`memory/lt_memory.py`) — durable facts in a local ChromaDB
store under `memory/semantic_memory/`, embedded with `bge-m3`. Similar
memories are injected into context ahead of every turn automatically, and
the agent can deliberately store and search facts through the
`save_memory` and `retrieve_memory` tools.
Configuration
All optional except `BASE_MODEL`:
Variable	What it controls
`BASE_MODEL`	the Ollama chat model
`SYSTEM_PROMPT`	the agent's system prompt
`AMADEUS_WORKSPACE`	relocates the directory the file tools are allowed to touch
`AMADEUS_TERMINAL_ACCESS`	terminal access level — see below
Terminal access
The agent can run shell commands on the host — one-shot with `run_command`, or
in a persistent bash session on a pty (`terminal_open`, `terminal_run`,
`terminal_read`, `terminal_send_keys`, `terminal_interrupt`, `terminal_list`,
`terminal_close`) that keeps its working directory and environment between
calls. How much it may do is set by `ACCESS_LEVEL` in
`src/amadeus_amth/tools/terminal_linux.py`, or by `AMADEUS_TERMINAL_ACCESS` in
`.env`:
Level	What it allows
`RESTRICTED`	An allowlist of read-only commands, run without a shell, inside the workspace only. No pipes, redirection or chaining.
`MEDIUM` (default)	A full shell anywhere on the filesystem, minus anything that administers the box: sudo, package installs, service/user/firewall/mount changes, containers, writes into system directories.
`UNRESTRICTED`	Everything except a short list of irreversible commands — wiping `/`, formatting a disk, powering off — which are refused at every level.
The levels are pattern checks on a command line: a guard rail against a
careless model, not a sandbox and not a security boundary. Anything determined
to get around them can (a variable, base64, a script file, an interpreter). The
real boundary has to come from the OS — run Amadeus as an unprivileged user
with no sudo rights, in a container, with nothing mounted it has no business
reaching.
