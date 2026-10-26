from dotenv import load_dotenv
from elevenlabs import ElevenLabs
import os
from pathlib import Path
import ollama


load_dotenv()

ollama = ollama.Client()


elevenlabs = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
voice_id = "AD29Onh88DukNe4gFB0V"


def auto_tag(text):


def gen_voice(text):
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id = voice_id,
        model_id = "eleven_v4",
        output_format = "mp3_44100_128"
    )

    output_path = Path(__file__).resolve().parents[2] / "voice" / "output.mp3"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("wb") as f:
        for chunk in audio:
            f.write(chunk)


if __name__ == "__main__":
    text = "how are you doing today?"
    gen_voice(text)


