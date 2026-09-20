import ollama
import requests
from fastapi import FastAPI
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
from .models import ChatRequest, HousekeepingPromptRequest, MaxIterationsRequest
from .tools.workspace_tool import WORKSPACE_ROOT

WEB_DIR = Path(__file__).parent / "web"
CHAT_LOG_DIR = WORKSPACE_ROOT / "chat_logs"

# Each turn starts with "[HH:MM] User: " or "[HH:MM] Amadeus: "; any line that
# doesn't is a continuation of a multi-line message.
TURN_HEADER = re.compile(r"^\[(\d{2}:\d{2})\] (User|Amadeus): ?(.*)$")


def today_log() -> Path:
    """Path to today's chat log, created if it doesn't exist yet. Resolved on
    every call so a server left running past midnight rolls over to a new file."""
    CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAT_LOG_DIR / f"{datetime.now():%Y-%m-%d}.txt"
    path.touch(exist_ok=True)
    return path


# The agent is supposed to make an empty reply impossible; this is the seatbelt
# for the day something upstream slips one through. A named notice tells the
# user what happened, where a blank bubble just looks like the app is broken.
BLANK_REPLY_NOTICE = "(Nothing came back from the model on that turn - please send it again.)"


def spoken(text: str) -> str:
    """Never let an empty string reach the UI or the chat log."""
    return text if (text or "").strip() else BLANK_REPLY_NOTICE


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
    # A turn whose text is blank renders as an empty bubble on reload, which
    # reads as a bug rather than as history. Drop those instead of replaying
    # them (a header line with the message on the lines below is not blank -
    # its continuations have already been folded in by this point).
    return [t for t in turns if t["text"].strip()]

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
            reply = spoken(await agent.chat(agent.housekeeping_prompt))
            append_log("Amadeus", f"(housekeeping beat) {reply}")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    append_log("User", request.message)
    reply = spoken(await app.state.agent.chat(request.message))
    append_log("Amadeus", reply)
    return {"response": reply}



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
