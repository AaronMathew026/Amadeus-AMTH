import ollama
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import contextlib
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .agent import (
    MAX_ITERATIONS_LIMIT,
    MIN_ITERATIONS,
    Amadeus,
    DEFAULT_HOUSEKEEPING_PROMPT,
)
from . import emotion, gen_voice, uploads
from .models import (
    ChatModeRequest,
    ChatRequest,
    HousekeepingPromptRequest,
    MaxIterationsRequest,
)
from .tools.workspace_tool import WORKSPACE_ROOT

WEB_DIR = Path(__file__).parent / "web"
CHAT_LOG_DIR = WORKSPACE_ROOT / "chat_logs"
VOICE_PATH = WORKSPACE_ROOT.parent / "voice" / "output.mp3"

# Each turn starts with "[HH:MM] User: " or "[HH:MM] Amadeus: "; any line that
# doesn't is a continuation of a multi-line message.
HOUSEKEEPING_MARK = "(housekeeping beat)"
TURN_HEADER = re.compile(r"^\[(\d{2}:\d{2})\] (User|Amadeus): ?(.*)$")


def today_log() -> Path:
    """Path to today's chat log, created if it doesn't exist yet. Resolved on
    every call so a server left running past midnight rolls over to a new file."""
    CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAT_LOG_DIR / f"{datetime.now():%Y-%m-%d}.txt"
    path.touch(exist_ok=True)
    return path


def append_log(speaker: str, text: str):
    with open(today_log(), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M}] {speaker}: {text}\n")


def read_log() -> list[dict]:
    turns = []
    with open(today_log(), "r", encoding="utf-8", errors="replace") as f:
        for line in f.read().splitlines():
            header = TURN_HEADER.match(line)
            if header:
                time, speaker, text = header.groups()
                turns.append({"role": "user" if speaker == "User" else "bot", "time": time, "text": text})
            elif turns:
                turns[-1]["text"] += "\n" + line
    # Heartbeat sweeps stay in the log file but never reach the chat UI.
    return [t for t in turns if HOUSEKEEPING_MARK not in t["text"]]

async def heartbeat(agent: Amadeus):
    while True:
        try:
            await asyncio.sleep(3600) # Wake up every 3600 seconds (hourly beat) - set this up as a configurable parameter later
            print("Heartbeat: Performing housekeeping tasks...")
            # Heartbeats never pass through the /chat endpoint, so without an
            # explicit append_log their turns vanish from the chat log (the
            # missing 08:29 sweep entry of 2026-09-15). Log the reply, marked
            # so it is never mistaken for a user-facing answer.
            # Read per beat, not once at startup: an edit from the settings
            # panel takes effect on the next sweep. Always the full agent,
            # whatever the panel's chat mode: the sweep files what it finds in
            # long-term memory, and the terminal mode has no tool for that.
            reply = await agent.chat(agent.housekeeping_prompt, speak=False, mode="full")
            append_log("Amadeus", f"{HOUSEKEEPING_MARK} {reply}")


            # await is used to ensure we dont block the event loop
            # Fill this in later
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Heartbeat error: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    ### Startup Tasks ###
     today_log()
     app.state.agent = Amadeus()
     task = asyncio.create_task(heartbeat(app.state.agent))
     yield # Signal that the application has started

     #### Shutdown Tasks ###
     task.cancel()
     with contextlib.suppress(asyncio.CancelledError):
         await task


app = FastAPI(lifespan=lifespan)
app.include_router(uploads.router)
app.include_router(emotion.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_chat(request: ChatRequest, on_event=None) -> dict:
    """One exchange, for /chat and /chat/stream alike: log it, answer it, and
    point at the reply's voice."""
    attached = "".join(f"\n📎 {name}" for name in request.files)
    append_log("User", request.message + attached)
    # Extracting an attachment's text (a PDF especially) is blocking work.
    message = await asyncio.to_thread(uploads.attach, request.message, request.files)
    reply = await app.state.agent.chat(message, speak=request.speak, on_event=on_event)
    append_log("Amadeus", reply)
    # Point at this reply's own clip, not output.mp3: when voice generation
    # fails or a path returns without speaking, output.mp3 still holds the
    # previous reply, and the UI would play that as if it were this one. The
    # clip is usually still being generated at this point — the text no
    # longer waits for its voice — and GET /voice/<id>.mp3 waits for it.
    audio = None
    if request.speak:
        clip_id = gen_voice.voice_id_for(reply)
        if gen_voice.has_clip(clip_id):
            audio = f"/voice/{clip_id}.mp3"
    return {"response": reply, "audio": audio}


@app.post("/chat")
async def chat(request: ChatRequest):
    return await run_chat(request)


# Exchanges running for /chat/stream. Each runs as its own task rather than
# inside the response, so a browser dropping the connection mid-reply (a phone
# locking, a tab closing) does not cancel it: the reply is still committed and
# logged, and turns up from /history on the next load. Held here because the
# event loop keeps only weak references to tasks.
_streaming = set()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """/chat as it happens, as newline-delimited JSON — one event per line:

    {"type": "delta", "text"}       more of the reply being written
    {"type": "replace", "text"}     the reply so far, rewritten from scratch
    {"type": "tools", "names"}      that turn ended in tool calls, now running;
                                    its text was not the reply
    {"type": "done", "response", "audio"}   the finished reply, as /chat returns it
    {"type": "error", "message"}    the exchange failed
    """
    events = asyncio.Queue()

    async def run():
        try:
            result = await run_chat(request, on_event=events.put_nowait)
        except Exception as e:
            print(f"[chat] Streamed exchange failed: {e!r}")
            events.put_nowait({"type": "error", "message": f"server error: {e!r}"})
        else:
            events.put_nowait({"type": "done", **result})

    task = asyncio.create_task(run())
    _streaming.add(task)
    task.add_done_callback(_streaming.discard)

    async def lines():
        while True:
            event = await events.get()
            yield json.dumps(event) + "\n"
            if event["type"] in ("done", "error"):
                return

    # A proxy in front (nginx and friends) buffers responses by default,
    # which would hold every line back until the reply was finished.
    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ReplayRequest(BaseModel):
    text: str


@app.post("/voice/replay")
async def voice_replay(request: ReplayRequest):
    """Audio for a reply the user wants to hear again. Clips of the last few
    spoken replies are cached, so those cost nothing; anything else is
    generated once and then cached too."""
    text = request.text.strip()
    if not text:
        return {"error": "Nothing to speak."}
    # A reply whose voice is still on its way joins that generation instead
    # of paying for a second one.
    clip_id = gen_voice.speak(text)
    if await gen_voice.clip_path(clip_id) is None:
        return {"error": "Voice generation failed."}
    return {"audio": f"/voice/{clip_id}.mp3"}


@app.get("/voice/output.mp3")
def voice_output():
    if not VOICE_PATH.is_file():
        return {"error": "No generated voice is available."}
    return FileResponse(VOICE_PATH, media_type="audio/mpeg")



@app.get("/voice/{clip_id}.mp3")
async def voice_clip(clip_id: str):
    if not re.fullmatch(r"[0-9a-f]{16}", clip_id):
        return {"error": "No generated voice is available."}
    # A reply's clip is asked for the moment its text arrives, usually before
    # the clip is finished; the request waits for it rather than failing.
    path = await gen_voice.clip_path(clip_id)
    if path is None:
        return {"error": "No generated voice is available."}
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/health")
def health():
    return {"message": "Amadeus is operational"}


def settings_payload() -> dict:
    """The whole settings state, so every write can hand the UI a fresh copy."""
    agent = app.state.agent
    return {
        "max_iterations": agent.max_iterations,
        "min_iterations": MIN_ITERATIONS,
        "max_iterations_limit": MAX_ITERATIONS_LIMIT,
        "housekeeping_prompt": agent.housekeeping_prompt,
        "default_housekeeping_prompt": DEFAULT_HOUSEKEEPING_PROMPT,
        # Lets the panel show "Reset" as already-done without diffing strings.
        "housekeeping_prompt_is_default": (
            agent.housekeeping_prompt == DEFAULT_HOUSEKEEPING_PROMPT
        ),
        "chat_mode": agent.chat_mode,
        # What each mode runs, so the panel can name the models.
        "chat_modes": {
            name: {"label": mode.label, "model": mode.model, "tools": len(mode.schemas)}
            for name, mode in agent.modes.items()
        },
    }


# Settings live in memory on the agent, so they last as long as the process and
# go back to the .env defaults on restart.
@app.get("/settings")
def get_settings():
    return settings_payload()


@app.post("/settings/max-iterations")
def set_max_iterations(request: MaxIterationsRequest):
    app.state.agent.change_iterations(request.max_iterations)
    return settings_payload()


@app.post("/settings/housekeeping-prompt")
def set_housekeeping_prompt(request: HousekeepingPromptRequest):
    app.state.agent.change_housekeeping_prompt(request.prompt)
    return settings_payload()


@app.post("/settings/housekeeping-prompt/reset")
def reset_housekeeping_prompt():
    app.state.agent.reset_housekeeping_prompt()
    return settings_payload()


@app.post("/settings/chat-mode")
def set_chat_mode(request: ChatModeRequest):
    app.state.agent.change_chat_mode(request.mode)
    return settings_payload()

@app.get("/history")
def history():
    """Today's conversation only, as structured turns for the web UI."""
    return {"turns": read_log()}


# Mounted last on purpose: a mount at "/" swallows every path that no route
# above it already claimed, so /chat and /health have to be declared first.
# html=True makes "/" serve web/index.html, which is the chat UI.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
