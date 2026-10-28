"""Emotion classifier for the sidebar kaomoji.

The UI asks for this *after* a reply is already on screen, so the extra beam
call never delays the answer. Any failure falls back to a neutral face.
"""

import ollama
from fastapi import APIRouter
from pydantic import BaseModel

# Must match the EMOTIONS names in web/kaomoji.js.
EMOTIONS = [
    "Happy", "Overjoyed", "Smiling", "Pouting", "Sad", "Crying", "Angry",
    "Surprised", "Embarrassed", "Sleepy", "In Love", "Winking", "Confused",
    "Cool", "Dizzy", "Hug",
]
DEFAULT_EMOTION = "Smiling"
MAX_CHARS = 1500

PROMPT = (
    "You pick the facial expression an assistant would make while saying the "
    "message it is given. Choose exactly one from this list: "
    + ", ".join(EMOTIONS)
    + ". Reply with the name only, nothing else."
)

client = ollama.AsyncClient()
router = APIRouter()


class EmotionRequest(BaseModel):
    text: str


async def classify(text: str) -> str:
    try:
        # Picking one name off a list needs little reasoning: on "low" beam
        # chose the same face as on its default effort in every test, in
        # half the time (1.7 s against 3.5 s on average).
        response = await client.chat(
            model="beam:cloud",
            think="low",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text[:MAX_CHARS]},
            ],
        )
        answer = (response.message.content or "").strip().lower()
    except Exception as e:
        print(f"[emotion] Classification failed: {e!r}")
        return DEFAULT_EMOTION
    # Longest names first so "overjoyed" is not mistaken for a shorter match.
    for name in sorted(EMOTIONS, key=len, reverse=True):
        if name.lower() in answer:
            return name
    return DEFAULT_EMOTION


@router.post("/emotion")
async def emotion(request: EmotionRequest):
    return {"emotion": await classify(request.text)}
