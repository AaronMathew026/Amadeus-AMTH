import ollama
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import contextlib
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
from .models import ChatRequest, HousekeepingPromptRequest, MaxIterationsRequest
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
            # panel takes effect on the next sweep.
            reply = await agent.chat(agent.housekeeping_prompt, speak=False)
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

@app.post("/chat")
async def chat(request: ChatRequest):
    attached = "".join(f"\n📎 {name}" for name in request.files)
    append_log("User", request.message + attached)
    reply = await app.state.agent.chat(
        uploads.attach(request.message, request.files), speak=request.speak
    )
    append_log("Amadeus", reply)
    # Point at this reply's own clip, not output.mp3: when voice generation
    # fails or a path returns without speaking, output.mp3 still holds the
    # previous reply, and the UI would play that as if it were this one.
    audio = None
    if request.speak:
        clip_id = gen_voice.voice_id_for(reply)
        if gen_voice.replay_path(clip_id).is_file():
            audio = f"/voice/{clip_id}.mp3"
    return {"response": reply, "audio": audio}


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
    clip_id = gen_voice.voice_id_for(text)
    if not gen_voice.replay_path(clip_id).is_file():
        try:
            await asyncio.to_thread(gen_voice.gen_voice, text)
        except Exception as e:
            print(f"[voice] Replay generation failed: {e!r}")
            return {"error": "Voice generation failed."}
    return {"audio": f"/voice/{clip_id}.mp3"}


@app.get("/voice/output.mp3")
def voice_output():
    if not VOICE_PATH.is_file():
        return {"error": "No generated voice is available."}
    return FileResponse(VOICE_PATH, media_type="audio/mpeg")



@app.get("/voice/{clip_id}.mp3")
def voice_clip(clip_id: str):
    path = gen_voice.replay_path(clip_id)
    if not re.fullmatch(r"[0-9a-f]{16}", clip_id) or not path.is_file():
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

@app.get("/history")
def history():
    """Today's conversation only, as structured turns for the web UI."""
    return {"turns": read_log()}


# Mounted last on purpose: a mount at "/" swallows every path that no route
# above it already claimed, so /chat and /health have to be declared first.
# html=True makes "/" serve web/index.html, which is the chat UI.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
