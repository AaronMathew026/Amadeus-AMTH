import ollama
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from models import ChatRequest

app = FastAPI()




@app.post("/chat")
def chat(request: ChatRequest):
    return {"response": f"Recieved message: {request.message} at {request.timestamp} in session {request.session_id}"}


@app.get("/")
def root():
    return {"message": "Amadeus is operational"}

