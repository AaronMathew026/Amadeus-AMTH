from dotenv import load_dotenv
from elevenlabs import ElevenLabs
import hashlib
import os
import shutil
from pathlib import Path
import ollama


load_dotenv()

ollama = ollama.Client()


elevenlabs = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
voice_id = "AD29Onh88DukNe4gFB0V"
TAGGING_PROMPT = r"""
You are an ElevenLabs audio-tagging assistant.

Rewrite the user's message for natural speech by inserting short audio-performance
tags where they improve delivery and by expanding technical shorthand into words
that a voice model can pronounce correctly. Preserve the original meaning and
all useful information. Do not add commentary, explanations, or new information.
Return only the final speech-ready message.

Speech formatting rules:
- Expand file extensions: ".md" becomes "Markdown file", ".py" becomes
    "Python file", and ".json" becomes "JSON file".
- Read times naturally: "3:30 PM" becomes "three thirty P M" and "14:05"
    becomes "fourteen oh five". Preserve the time and AM/PM information.
- Read dates, numbers, percentages, currencies, and measurements naturally
    without changing their values. For example, "25%" becomes "twenty-five
    percent" and "$10.50" becomes "ten dollars and fifty cents".
- Expand common technical shorthand when needed: "API" becomes "A P I",
    "URL" becomes "U R L", and "HTTP" becomes "H T T P".
- Make URLs, email addresses, paths, command flags, and code readable aloud
    without inventing content. For example, "docs/api/v1" may become
    "docs slash A P I slash v one".
- Read LaTeX math as spoken mathematics and drop the dollar signs and
    backslashes: "$T(n) \le c\lg n$" becomes "T of n is less than or equal to
    c log base two of n", and "\lceil n/2 \rceil" becomes "the ceiling of n
    over two".
- Keep product names, proper nouns, code identifiers, and quoted text intact
    unless changing only their spoken pronunciation is necessary.
- Do not spell out ordinary words, and do not convert the message into Markdown,
    JSON, a list, or an explanation.

Use only the following tags. Do not invent, modify, or combine tags:

High energy: [excited] [playful] [amazed] [powerful] [proud] [optimistic] [startled]
Heated: [mad] [aggressive] [bitter] [critical] [repelled]
Tense: [anxious] [stressed] [scared] [threatened] [vulnerable] [confused] [busy]
Low energy: [tired] [bored] [distant] [despair] [let down]
Calm: [peaceful] [content] [curious] [thoughtful] [trusting]
Other emotions: [sad] [angry] [happily] [sorrowful] [sarcastic] [mischievously]
[jittery] [smug] [disgusted] [hopeful] [nervous] [triumphant]
Delivery / volume: [whispers] [shouts] [softly] [quietly] [hushed]
[barely audible] [booming] [low, threatening]
Pacing: [slowly] [rushed] [pause] [long pause] [drawn out] [snappy] [speedy]
Reactions: [laughs] [laughs harder] [starts laughing] [chuckles] [wheezing]
[sighs] [exhales] [gasps] [clears throat] [crying] [snorts]
Accents / characters: [British accent] [French accent] [Australian accent]
[Irish accent] [pirate voice] [strong X accent]
Sound effects: [gunshot] [applause] [clapping] [explosion] [swallows] [gulps]
[thunder rumbling] [footsteps] [door creaking] [dog barking]
Experimental: [sings] [woo] [fart]

Place each tag immediately before the words it controls. Use no more than five tags and do not force a tag when the tone is already clear. Experimental tags
may be inconsistent across voices, so use them only when clearly appropriate.
Tags must never contain instructions, markup, JSON, or commentary.

Examples:

Input: I am glad you told me. We can work through this together.
Output: [warmly] I am glad you told me. [calmly] We can work through this together.

Input: That is fantastic news!
Output: [excited] That is fantastic news!

Input: Please restart the service and try again.
Output: [calmly] Please restart the service and try again.

Input: What would you like to do next?
Output: What would you like to do next?

Input: The report is in summary.md and the meeting starts at 3:30 PM.
Output: The report is in the Markdown file summary, and the meeting starts at three thirty P M.

Input: Send 25% of the $10.50 payment through the API at https://example.com/api.
Output: Send twenty-five percent of the ten dollars and fifty cents payment through the A P I at H T T P S colon slash slash example dot com slash A P I.

Input: Run python app.py --port 8080.
Output: Run Python app dot P Y dash dash port eight zero eight zero.

Your entire response must be the final tagged message and nothing else.
"""

def auto_tag(text):
    response = ollama.chat(
        model="beam:cloud",
        messages = [
            {"role": "system", "content": TAGGING_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    tagged_text = response.message.content.strip()
    print(f"Auto-tagged text: {tagged_text}")
    return tagged_text or text



VOICE_DIR = Path(__file__).resolve().parents[2] / "voice"
# Replies whose audio is kept for replay; older ones are deleted.
KEEP_REPLAYS = 5


def voice_id_for(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def replay_path(clip_id):
    return VOICE_DIR / f"{clip_id}.mp3"


def _prune():
    clips = sorted(VOICE_DIR.glob("????????????????.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in clips[KEEP_REPLAYS:]:
        old.unlink(missing_ok=True)


def gen_voice(text):
    """Speak `text`, keep the clip for replay, and return its id."""
    text_id = voice_id_for(text)
    spoken = auto_tag(text)
    audio = elevenlabs.text_to_speech.convert(
        text=spoken,
        voice_id = voice_id,
        model_id = "eleven_v4_turbo",
        output_format = "mp3_44100_128"
    )

    VOICE_DIR.mkdir(exist_ok=True)
    clip = replay_path(text_id)
    with clip.open("wb") as f:
        for chunk in audio:
            f.write(chunk)
    shutil.copyfile(clip, VOICE_DIR / "output.mp3")
    _prune()
    return text_id


if __name__ == "__main__":
    text = """Hey twin, how are you doing?"""
    gen_voice(text)


