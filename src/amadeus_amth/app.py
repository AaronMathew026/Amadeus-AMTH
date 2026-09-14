import ollama
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .agent import Amadeus
from .models import ChatRequest

WEB_DIR = Path(__file__).parent / "web"

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
    reply = await app.state.agent.chat(request.message)
    return {"response": reply}


@app.get("/health")
def health():
    return {"message": "Amadeus is operational"}


# Mounted last on purpose: a mount at "/" swallows every path that no route
# above it already claimed, so /chat and /health have to be declared first.
# html=True makes "/" serve web/index.html, which is the chat UI.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
