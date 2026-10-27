from pydantic import BaseModel, Field

from .agent import MAX_ITERATIONS_LIMIT, MIN_ITERATIONS


class ChatRequest(BaseModel):
    message: str
    # Names returned by POST /upload for files attached to this message.
    files: list[str] = []
    # False when voice is switched off in the UI: no TTS (ElevenLabs) calls at all.
    speak: bool = True


class MaxIterationsRequest(BaseModel):
    # Bounded here rather than in the endpoint so an out-of-range value comes
    # back as a 422 with a readable message instead of silently sticking.
    max_iterations: int = Field(ge=MIN_ITERATIONS, le=MAX_ITERATIONS_LIMIT)


class HousekeepingPromptRequest(BaseModel):
    # min_length keeps an empty box from turning the heartbeat into a beat that
    # sends the model nothing to act on.
    prompt: str = Field(min_length=1)
