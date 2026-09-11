import ollama
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from models import ChatRequest

app = FastAPI()




@app.post("/chat")
def chat(request: ChatRequest):
    response = ollama.chat(
        model="gemma4:31b-cloud",
        messages=[{"role": "system", "content": "You are Amadeus, a large language model trained by Ollama. You are helpful, creative, clever, and very helpful."}, {"role": "user", "content": request.message}],
    )
    return {"response": response}


@app.get("/")
def root():
    return {"message": "Amadeus is operational"}