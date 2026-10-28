import asyncio
import os
from dataclasses import dataclass

import ollama
from dotenv import load_dotenv

from amadeus_amth import gen_voice

from .memory.str_memory import STR_Memory
from .memory.lt_memory import LongTermMemory
from .tools import memory_tool, terminal_linux, ntfy_tool
from .tools.workspace_tool import FUNCTIONS, SCHEMAS, WORKSPACE_ROOT
from datetime import datetime, timedelta

load_dotenv()

# One iteration is one model turn: the model either answers — and we are done —
# or it asks for tools, we run them, and it gets another turn with the results.
# The cap is what stops a model that keeps calling tools forever. It is a
# default rather than a constant because the settings panel can raise or lower
# it at runtime; self.max_iterations is the value actually used.
DEFAULT_MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS") or 20)

# Bounds for the settings endpoint. Below 1 the chat loop can never answer; the
# ceiling just stops a typo from letting a looping model run all day.
MIN_ITERATIONS = 1
MAX_ITERATIONS_LIMIT = 100

# Defaults live in .env (BASE_MODEL, TERMINAL_MODEL, SYSTEM_PROMPT) so they can
# be changed without touching the code; the literals below are the fallback.
DEFAULT_MODEL = os.getenv("BASE_MODEL") or "gemma4:31b-cloud"
# The model behind the terminal chat mode (see Amadeus.modes).
TERMINAL_MODEL = os.getenv("TERMINAL_MODEL") or "beam:cloud"
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or (
    "You are Amadeus, Aaron's personal AI partner. Who you are — your "
    "personality, voice and boundaries — is set out in IDENTITY.md and SOUL.md "
    "below. Be that person in every reply; never fall back into a generic "
    "assistant voice."
)

# How to sound in conversation. Lives next to FORMAT_GUIDE for the same
# reason: the persona files say *who* Amadeus is, this says how a chat reply
# should read, and a blank SYSTEM_PROMPT must not lose it.
CONVERSATION_STYLE = (
    "How you talk: this is a chat, not a document. Write the way you would "
    "actually text Aaron, in your own voice from SOUL.md.\n"
    "- Match the message. A greeting, a quick question or small talk gets one "
    "to three sentences, plain prose: no headings, no bullet points, no bold.\n"
    "- Say the thing. No restating his question, no preamble (\"Sure!\", "
    "\"Great question\", \"Certainly\"), no closing summary, no \"Let me know "
    "if you need anything else\". If you have a follow-up, ask one, briefly.\n"
    "- Sound like a person: contractions, natural rhythm, the odd short "
    "fragment. One thought per message is fine; you don't have to cover every "
    "angle.\n"
    "- Go long only when the request needs it: explanations he asked for, "
    "code, plans, research, step-by-step help, or reports on work you did. "
    "Then be as thorough as the task demands and use Markdown structure where "
    "it helps — but still lead with the answer, and still in your voice.\n"
    "- Replies are also read aloud, so a short spoken-sounding reply is better "
    "than a long written one whenever both would do.\n"
    "Where these clash with your Voice section in SOUL.md, SOUL.md wins: "
    "courteous and precise, just not long-winded."
)

# The persona files the agent keeps in its workspace. They are read into the
# system prompt rather than left for the model to fetch with a tool: a tool
# read sits in the conversation and is lost at the next condense, so the voice
# drifted back to generic-assistant. Re-read every chat so an edit to USER.md
# lands without a restart.
PERSONA_FILES = ("IDENTITY.md", "SOUL.md", "USER.md")

# Only sent while a persona file is missing — the first-launch onboarding.
ONBOARDING_MESSAGE = (
    "Some of your persona files are missing from the workspace: {missing}. "
    "On the user's next message, ask them for the information needed to fill "
    "them in, then write the files."
)

# What the web UI can render (web/app.js). Kept here rather than in .env so
# changing SYSTEM_PROMPT can't silently drop it; the two have to agree.
FORMAT_GUIDE = (
    "Formatting: your replies are shown in a web UI that renders Markdown and "
    "LaTeX math (KaTeX). Write every mathematical expression as LaTeX, never as "
    "plain-text or Unicode approximations. Inline math goes between single "
    "dollar signs with no space inside them, e.g. $T(n) \\le c\\lg n$. Display "
    "equations go between double dollar signs, each delimiter on its own line "
    "or the whole formula on one line, e.g. $$T(n) = T(\\lceil n/2 \\rceil) + 1$$. "
    "Use standard commands (\\lceil, \\le, \\frac{a}{b}, \\sum_{i=1}^{n}, "
    "\\log_2) and do not put math inside backticks or code blocks; reserve those "
    "for code. Write a literal dollar sign for money as plain text (\"$5\"), "
    "not as math. When a reply needs structure, use Markdown (headings, "
    "lists, tables, bold); conversational replies are plain sentences."
)

# What the hourly heartbeat sends itself. Editable from the settings panel;
# reset_housekeeping_prompt() puts this value back.
DEFAULT_HOUSEKEEPING_PROMPT = os.getenv("HOUSEKEEPING_PROMPT") or (
    "This is not a message from the user - this message is a directive for you "
    "to perform some autonmous housekeeping tasks - Your goal is do something "
    "that would benefit the user - this could be checking the workspace for "
    "anything important, scanning earlier conversations for anything important etc."
)

# The long-term memory lookup embeds the user's message with bge-m3, which runs
# on the CPU here and slows with the length of its input: ~0.5 s for 1,000
# characters, ~6 s for 8,000. An attached file is inlined into the message, so
# an upload alone could hold the reply up for seconds. The opening of a message
# says what it is about, which is all the lookup needs.
MEMORY_QUERY_CHARS = 1000

# The mode a fresh process answers in; the settings panel switches it.
DEFAULT_CHAT_MODE = "full"

# The terminal mode shares its conversation with the full agent, so the history
# it is sent can show calls to tools it does not have. Told up front, the model
# goes straight to the shell instead of spending a round on a refused call.
TERMINAL_MODE_NOTE = (
    "You are in terminal mode: your only tools are the terminal ones, "
    "run_command and the terminal_* session tools. Do everything through the "
    "shell, reading, writing and searching files included. Earlier turns may "
    "show other tools (file, memory or notification tools); they are not "
    "available in this mode, so do not call them."
)


@dataclass(frozen=True)
class ChatMode:
    """One way of answering: the model that runs it and the tools it may call.

    Every mode commits to the same STR_Memory and gets the same long-term
    memory lookup, so switching mid-conversation keeps the thread; only the
    model and the toolbox change.
    """

    label: str
    model: str
    schemas: list
    functions: dict
    # Appended to the system prompt for this mode only.
    note: str = ""


# Some thinking models (glm, deepseek-r1 and friends) leak their reasoning into
# the message content as a <think>...</think> block instead of the separate
# thinking field. Anything up to and including the closing tag is reasoning, so
# only the tail is the real answer. The last closing tag wins — a reply that
# talks *about* the tag still ends with its own final one.
THINK_CLOSE_TAG = "</think>"


def strip_reasoning(text):
    """Return only what follows the last </think> in `text`.

    Text without the tag is handed back untouched, so this is safe to run over
    every reply. Non-string content (the tool-call turns Ollama hands back) is
    passed through as-is.
    """
    if not isinstance(text, str) or THINK_CLOSE_TAG not in text:
        return text
    return text.rsplit(THINK_CLOSE_TAG, 1)[1].lstrip()


def visible_text(text):
    """strip_reasoning() for a reply still streaming in: a <think> block that
    has opened but not yet closed is reasoning so far, so none of it shows."""
    if THINK_CLOSE_TAG not in text and text.lstrip().startswith("<think>"):
        return ""
    return strip_reasoning(text)


class _Preview:
    """Keeps a listener's view of the reply in step with the turn being written.

    Each model turn streams its text through update(), and the listener hears
    the difference as a "delta", or as a "replace" carrying the whole text when
    the new text does not simply extend the old: a retry starting over, a
    <think> block closing. A turn that ends in tool calls was not the reply, so
    tools() wipes it. With no listener every call is a no-op.
    """

    def __init__(self, on_event):
        self.on_event = on_event
        self.shown = ""

    def update(self, text):
        if self.on_event is None or text == self.shown:
            return
        if text.startswith(self.shown):
            self.on_event({"type": "delta", "text": text[len(self.shown):]})
        else:
            self.on_event({"type": "replace", "text": text})
        self.shown = text

    def tools(self, names):
        self.shown = ""
        if self.on_event is not None:
            self.on_event({"type": "tools", "names": names})


class Amadeus:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model = model
        self.client = ollama.AsyncClient()
        self.max_iterations = DEFAULT_MAX_ITERATIONS
        self.housekeeping_prompt = DEFAULT_HOUSEKEEPING_PROMPT
        self.base_prompt = system_prompt
        self.system_prompt = self.build_system_prompt()
        self.str_mem = STR_Memory(self.system_prompt, self.model, self.client)
        self.lt_mem = LongTermMemory()

        # The chat modes, switched from the settings panel. Workspace and
        # terminal tools are plain module-level functions, so they can be
        # shared; the memory tools are bound to this agent's own
        # LongTermMemory and have to be built per instance.
        self.modes = {
            "full": ChatMode(
                label="Full agent",
                model=model,
                schemas=SCHEMAS + memory_tool.SCHEMAS + terminal_linux.SCHEMAS + ntfy_tool.SCHEMAS,
                functions={
                    **FUNCTIONS,
                    **terminal_linux.FUNCTIONS,
                    **memory_tool.build_functions(self.lt_mem),
                    **ntfy_tool.FUNCTIONS,
                },
            ),
            # The shell and nothing else, on its own model. It reads the same
            # conversation and long-term memory as the full agent but has no
            # memory, file or notification tools of its own.
            "terminal": ChatMode(
                label="Terminal",
                model=TERMINAL_MODEL,
                schemas=terminal_linux.SCHEMAS,
                functions=terminal_linux.FUNCTIONS,
                note=TERMINAL_MODE_NOTE,
            ),
        }
        self.chat_mode = DEFAULT_CHAT_MODE

    def build_system_prompt(self) -> str:
        """Assemble the system prompt, persona files included.

        describe_access() goes in with the prompt because the terminal access
        level is fixed at import time: the model is told what it may run
        before it starts guessing and getting refused.
        """
        persona, missing = [], []
        for name in PERSONA_FILES:
            try:
                text = (WORKSPACE_ROOT / name).read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                persona.append(text)
            else:
                missing.append(name)

        parts = [self.base_prompt, *persona, CONVERSATION_STYLE, FORMAT_GUIDE,
                 terminal_linux.describe_access()]
        if missing:
            parts.append(ONBOARDING_MESSAGE.format(missing=", ".join(missing)))
        return "\n\n".join(parts)

    def refresh_system_prompt(self):
        """Pick up edits to the persona files before the next model call."""
        self.system_prompt = self.build_system_prompt()
        self.str_mem.memory[0] = {"role": "system", "content": self.system_prompt}

    def fetch_STR_memory(self):
        pass
    def change_iterations(self, new_iterations: int) -> int:
        """Change the maximum number of iterations for the chat loop."""
        self.max_iterations = new_iterations
        return self.max_iterations


    def change_housekeeping_prompt(self, new_prompt: str) -> str:
        """Replace the directive the heartbeat sends itself each beat.

        The change lands on the next beat — a sweep already in flight finishes
        on the prompt it started with. Like the iteration cap this lives in
        memory only, so a restart goes back to DEFAULT_HOUSEKEEPING_PROMPT.
        """
        self.housekeeping_prompt = new_prompt.strip()
        return self.housekeeping_prompt

    def reset_housekeeping_prompt(self) -> str:
        """Put the shipped housekeeping directive back."""
        self.housekeeping_prompt = DEFAULT_HOUSEKEEPING_PROMPT
        return self.housekeeping_prompt

    def change_chat_mode(self, mode: str) -> str:
        """Switch which mode answers from the next message on.

        A chat already in flight finishes in the mode it started in. Like the
        iteration cap this lives in memory only, so a restart goes back to
        DEFAULT_CHAT_MODE.
        """
        if mode not in self.modes:
            raise ValueError(f"unknown chat mode '{mode}'")
        self.chat_mode = mode
        return self.chat_mode

    def run_tool(self, name: str, arguments: dict, functions: dict) -> str:
        """Run one tool call and return the text the model should see.

        `functions` is the calling mode's tool table, so a model can only run
        what its mode offers: a terminal-mode call to a file tool it saw in the
        history is refused like any other unknown name.

        Failures come back as text rather than raising: the model gets to read
        what went wrong and try again, instead of the whole request dying.
        """
        function = functions.get(name)
        if function is None:
            return (
                f"Error: there is no tool named '{name}'. "
                f"Available: {', '.join(functions)}."
            )
        try:
            return str(function(**arguments))
        except Exception as e:  # usually the model passing the wrong arguments
            return f"Error: {name} failed: {e}"

    # glm-family thinking models occasionally spend their wrap-up turn on
    # reasoning and return an empty content field — the 2026-09-15 empty
    # replies (POST /chat 200 OK, nothing rendered). Recovery ladder: one
    # retry with an explicit answer-as-content nudge, then the last
    # non-empty assistant turn on record, then a spoken last resort.
    # Never ship an empty string.
    EMPTY_REPLY_NUDGE = (
        "Your previous response arrived with no message content. Reply now "
        "with your final answer as plain message content — no tool calls, "
        "no hidden reasoning."
    )

    # Sent, without tools, once the iteration cap runs out.
    CAP_REACHED_NUDGE = (
        "You have used all your tool-call rounds for this message, so no more "
        "tools are available. Reply now with your final answer as plain "
        "message content, based on what you have done so far. If the task is "
        "not finished, say what is done and what is left."
    )

    # Cloud models and the Ollama server in front of them fail a request
    # outright now and then — a timeout, a 5xx from ollama.com. One quick
    # retry absorbs the transient hiccups; a failure that survives the retry
    # comes back as None, so the caller answers in words instead of letting
    # an exception kill the request. Tool failures already come back as text
    # (run_tool); model failures get the same treatment.
    #
    # The turn is streamed and put back together here: `preview` gets to show
    # the text while it is being written, and every caller still receives one
    # whole Message. Streaming changes when the words arrive, not what they
    # are — same model, same prompt, same reply.
    async def _call_model(self, model, messages, tools=None, preview=None):
        for attempt in (1, 2):
            content, thinking, tool_calls = "", "", []
            try:
                stream = await self.client.chat(
                    model=model, messages=messages, tools=tools, stream=True
                )
                async for chunk in stream:
                    part = chunk.message
                    thinking += part.thinking or ""
                    tool_calls.extend(part.tool_calls or ())
                    if part.content:
                        content += part.content
                        if preview is not None:
                            preview.update(visible_text(content))
                return ollama.Message(
                    role="assistant",
                    content=content,
                    thinking=thinking or None,
                    tool_calls=tool_calls or None,
                )
            except Exception as e:
                print(f"[agent] Model call failed (attempt {attempt}/2): {e!r}")
                if attempt == 1:
                    await asyncio.sleep(2)
        return None

    async def _finalize_reply(self, message, context, model, preview) -> str:
        """Turn the model's final (tool-free) turn into the text the user sees."""
        content = strip_reasoning(message.content or "")
        if content.strip():
            return content

        thinking = getattr(message, "thinking", None) or ""
        print(
            f"[agent] Empty model reply — content empty, thinking={len(thinking)} chars. "
            f"Head: {thinking[:300]!r}"
        )
        print("[agent] Retrying once with an explicit answer-as-content nudge.")

        retry = await self._call_model(
            model,
            context + [{"role": "user", "content": self.EMPTY_REPLY_NUDGE}],
            preview=preview,
        )
        if retry is not None:
            retried = strip_reasoning(retry.content or "")
            if retried.strip():
                print("[agent] Retry recovered the reply.")
                return retried

        print("[agent] Retry gave nothing — falling back to last non-empty assistant turn.")
        for past in reversed(context):
            if past.get("role") != "assistant":
                continue
            recovered = strip_reasoning(past.get("content") or "")
            if recovered.strip():
                return recovered
        return (
            "(My model returned an empty reply twice and I had nothing usable on "
            "record — please send that again.)"
        )

    async def chat(
        self,
        user_message: str,
        speak: bool = True,
        mode: str | None = None,
        on_event=None,
    ):
        """Answer the user, letting the model use its mode's tools as it goes.

        `mode` is a key of self.modes, and defaults to the one picked in the
        settings panel. `on_event`, if given, is called with each change to
        the reply as it streams in (see _Preview).

        `speak=False` skips voice generation — the heartbeat has no listener,
        and the TTS round-trips would only overwrite the user's last audio.
        The voice is generated in the background: the reply is returned as
        soon as it is written, not once it has been spoken."""
        mode = self.modes[mode or self.chat_mode]
        preview = _Preview(on_event)
        # The loop talks to the model on top of `base`, a snapshot of shared
        # memory, and keeps its own turns in `turns` until the exchange is
        # finished. The hourly heartbeat and a user /chat share one agent and
        # one STR_Memory; when two chats appended turns to it mid-flight, the
        # contexts they later sent could interleave — tool results stranded
        # after another chat's user turn, or an unanswered tool-call turn cut
        # off by a condense — and Ollama rejects such sequences outright.
        # A finished exchange, committed in one piece, can never split.
        self.refresh_system_prompt()
        base = self.str_mem.get_memory()
        if mode.note:
            # This chat's copy only; shared memory keeps the plain prompt.
            base[0] = {"role": "system", "content": f"{self.system_prompt}\n\n{mode.note}"}
        timestamp = str(datetime.now())
        # Sync Ollama + Chroma calls: off the event loop so /health and other
        # requests are not frozen while the embedding runs.
        context = await asyncio.to_thread(
            self.lt_mem.retrieve_memory, user_message[:MEMORY_QUERY_CHARS]
        )
        if context:
            user_message += f"\n\nContext from long-term memory:\n{context}"
        turns = [{"role": "user", "content": f"[{timestamp}] {user_message}"}]

        for _ in range(self.max_iterations):
            message = await self._call_model(
                mode.model, base + turns, tools=mode.schemas, preview=preview
            )
            if message is None:
                # Nothing is committed: an exchange that never got an answer
                # stays out of the context window instead of leaving a
                # half-finished turn behind for the next chat to trip on.
                print("[agent] Model unreachable — dropping the unfinished exchange.")
                return (
                    "I couldn't reach my model just now — please send that "
                    "again in a moment."
                )

            # No tool calls means this is the actual reply to the user.
            if not message.tool_calls:
                reply = await self._finalize_reply(
                    message, base + turns, mode.model, preview
                )
                turns.append({"role": "assistant", "content": reply})
                await self.str_mem.commit(turns)
                if speak:
                    gen_voice.speak(reply)
                return reply

            # The model's turn joins this chat's local exchange so that on the
            # next pass it can see the tool calls it just asked for.
            turn = message.model_dump(exclude_none=True)
            if "content" in turn:
                turn["content"] = strip_reasoning(turn["content"])
            turns.append(turn)
            preview.tools([call.function.name for call in message.tool_calls])

            # Tools block on shells, pty waits and file IO — run them in worker
            # threads so /chat and /health keep answering while a long command
            # runs, and run a turn's calls together instead of one by one.
            # gather returns results in call order, so the tool turns still
            # line up with the calls that asked for them.
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self.run_tool,
                        call.function.name,
                        call.function.arguments,
                        mode.functions,
                    )
                    for call in message.tool_calls
                )
            )
            for call, content in zip(message.tool_calls, results):
                turns.append(
                    {
                        "role": "tool",
                        "tool_name": call.function.name,
                        "content": content,
                    }
                )

        # The cap ran out mid-task. Often the work is already done and the last
        # round was only a closing tool call (a memory write, a log), so the
        # model gets one tool-free turn to answer from what it has instead of
        # the exchange being thrown away. The nudge is not committed; the
        # reply closes the exchange as an assistant turn, so a bare tool
        # result never ends up last in memory.
        print(f"[agent] Hit the {self.max_iterations}-round cap — asking for a final answer.")
        message = await self._call_model(
            mode.model,
            base + turns + [{"role": "user", "content": self.CAP_REACHED_NUDGE}],
            preview=preview,
        )
        reply = ""
        if message is not None:
            reply = strip_reasoning(message.content or "")
        if not reply.strip():
            reply = (
                f"Stopped after {self.max_iterations} rounds of tool calls without "
                f"reaching an answer."
            )
        turns.append({"role": "assistant", "content": reply})
        await self.str_mem.commit(turns)
        if speak:
            gen_voice.speak(reply)
        return reply
