import ollama
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import contextlib
from contextlib import asynccontextmanager
import uvicorn

from agent import Amadeus
from models import ChatRequest




async def heartbeat(agent: Amadeus):
    while True:
        try:
            await asyncio.sleep(60) # Wake up every 60 seconds - set this up as a configurable parameter later
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



@app.post("/chat")
async def chat(request: ChatRequest):
    reply = await app.state.agent.chat(request.message)
    return {"response": reply}


@app.get("/")
def root():
    return {"message": "Amadeus is operational"}


def main() -> None:
    uvicorn.run("main:app", host="127.0.0.2", port=8000, reload=False)


if __name__ == "__main__":
    main()