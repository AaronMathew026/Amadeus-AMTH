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
from .agent import Amadeus
from .models import ChatRequest
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
    return turns

house_keeping_prompt =  "This is not a message from the user - this message is a directive for you to perform some autonmous housekeeping tasks - Your goal is do something that would benefit the user - this could be checking the workspace for anything important, scanning earlier conversations for anything important etc."


async def heartbeat(agent: Amadeus):
    while True:
        try:
            await asyncio.sleep(600) # Wake up every 60 seconds - set this up as a configurable parameter later
            print("Heartbeat: Performing housekeeping tasks...")
            await agent.chat(house_keeping_prompt) # Call the chat function to perform housekeeping tasks - this will allow the model to perform any necessary housekeeping tasks


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
    reply = await app.state.agent.chat(request.message)
    append_log("Amadeus", reply)
    return {"response": reply}



@app.get("/health")
def health():
    return {"message": "Amadeus is operational"}

@app.get("/history")
def history():
    """Today's conversation only, as structured turns for the web UI."""
    return {"turns": read_log()}


# Mounted last on purpose: a mount at "/" swallows every path that no route
# above it already claimed, so /chat and /health have to be declared first.
# html=True makes "/" serve web/index.html, which is the chat UI.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
