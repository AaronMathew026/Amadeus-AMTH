"""Amadeus — a personal AI assistant served over a small FastAPI app."""

import uvicorn


def main() -> None:
    uvicorn.run("amadeus_amth.app:app", host="127.0.0.2", port=8000, reload=False)
