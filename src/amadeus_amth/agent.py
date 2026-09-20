import asyncio
import os

import ollama
from dotenv import load_dotenv

from .memory.str_memory import STR_Memory
from .memory.lt_memory import LongTermMemory
from .tools import memory_tool, terminal_linux
from .tools.workspace_tool import FUNCTIONS, SCHEMAS
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

# How many times a turn that came back as reasoning-only may be asked again
# for a real message. Each one is a whole model round-trip, so this is the
# ceiling on how long a user waits for a reply that never arrives.
EMPTY_REPLY_RETRIES = int(os.getenv("EMPTY_REPLY_RETRIES") or 3)

# Defaults live in .env (BASE_MODEL, SYSTEM_PROMPT) so they can be changed
# without touching the code; the literals below are the fallback.
DEFAULT_MODEL = os.getenv("BASE_MODEL") or "gemma4:31b-cloud"
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or (
    "You are Amadeus, a personal AI assistant created by Aaron. You are a large "
    "language model that can answer questions, provide information, and assist "
    "with various tasks. You are knowledgeable in a wide range of topics and can "
    "provide helpful and accurate responses. You are also capable of engaging in "
    "casual conversation and providing entertainment. Your goal is to be helpful, "
    "informative, and engaging for the user."
)



def _split_models(raw: str) -> list[str]:
    """Parse a comma-separated model list from .env, dropping blanks."""
    return [name.strip() for name in (raw or "").split(",") if name.strip()]


# The catalogue the settings panel offers. It lives in .env as one comma-
# separated line so adding a model is an edit to the environment rather than to
# the code; anything Ollama will accept as a model name belongs here.
DEFAULT_CLOUD_MODELS = _split_models(os.getenv("CLOUD_MODELS")) or [DEFAULT_MODEL]

# Three named slots rather than a free-form model box: the switch beside the
# chat composer flips between them mid-conversation, so a quick question can go
# to something cheap and a hard one to something big without opening settings.
TIERS = ("low", "medium", "high")
DEFAULT_MODEL_TIERS = {
    tier: os.getenv(f"{tier.upper()}_MODEL") or DEFAULT_MODEL for tier in TIERS
}

_boot_tier = (os.getenv("DEFAULT_TIER") or "medium").strip().lower()
# A typo in .env should not take the server down with a KeyError on the first
# message, so an unrecognised tier just means the middle one.
DEFAULT_TIER = _boot_tier if _boot_tier in TIERS else "medium"

# What the hourly heartbeat sends itself. Editable from the settings panel;
# reset_housekeeping_prompt() puts this value back.
DEFAULT_HOUSEKEEPING_PROMPT = os.getenv("HOUSEKEEPING_PROMPT") or (
    "This is not a message from the user - this message is a directive for you "
    "to perform some autonmous housekeeping tasks - Your goal is do something "
    "that would benefit the user - this could be checking the workspace for "
    "anything important, scanning earlier conversations for anything important etc."
)


class Amadeus:
    def __init__(
        self,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        startup_message: str = "Your main server is running. This may the first time you have been launched. On the next user message check  for the Following files. IDENTITY.md,SOUL.md,USER.md, once the user sends their message, check if these files exist, if they do, continue as if you already know them. If these files are mssing, promptly ask them for the information required to fill the files in. THIS IS A CRITICAL MESSAGE, DO NOT FORGET IT.",

    ):
        # The three slots and which one is live. The model actually used is
        # whatever sits in the active slot, so the switch in the UI only has to
        # move `active_tier`. An explicit `model` argument overrides that
        # slot — constructing Amadeus(model="x") still runs "x".
        self.model_tiers = dict(DEFAULT_MODEL_TIERS)
        self.active_tier = DEFAULT_TIER
        if model:
            self.model_tiers[self.active_tier] = model
        self.model = self.model_tiers[self.active_tier]
        self.client = ollama.AsyncClient()
        self.max_iterations = DEFAULT_MAX_ITERATIONS
        self.housekeeping_prompt = DEFAULT_HOUSEKEEPING_PROMPT
        # describe_access() goes in with the prompt because the terminal access
        # level is fixed at import time: the model is told what it may run
        # before it starts guessing and getting refused.
        self.system_prompt = "\n\n".join(
            [system_prompt, terminal_linux.describe_access(), startup_message]
        )
        self.str_mem = STR_Memory(self.system_prompt, self.model, self.client)
        self.lt_mem = LongTermMemory()
        # One turn at a time. chat() interleaves its own messages with tool
        # results across several model round-trips, and the hourly heartbeat
        # calls chat() on this same agent and short-term memory. Without this
        # lock the two transcripts braid together: a `tool` message can end up
        # separated from the assistant turn that asked for it, and a model
        # handed that shape tends to answer with nothing at all.
        self._turn_lock = asyncio.Lock()

        # The tool registry. Workspace and terminal tools are plain
        # module-level functions, so they can be shared; the memory tools are
        # bound to this agent's own LongTermMemory and have to be built per
        # instance.
        self.schemas = SCHEMAS + memory_tool.SCHEMAS + terminal_linux.SCHEMAS
        self.functions = {
            **FUNCTIONS,
            **terminal_linux.FUNCTIONS,
            **memory_tool.build_functions(self.lt_mem),
        }

    def fetch_STR_memory(self):
        pass
    def change_iterations(self, new_iterations: int) -> int:
        """Change the maximum number of iterations for the chat loop."""
        self.max_iterations = new_iterations
        return self.max_iterations
    def change_model(self, new_model: str) -> str:
        """Run every following turn on `new_model`.

        Short-term memory keeps its own copy of the name for the condense call,
        so it is repointed here as well - otherwise a switch leaves the summary
        pass still talking to the model the user just moved off.
        """
        new_model = new_model.strip()
        self.model = new_model
        self.str_mem.model = new_model
        return self.model

    def _check_tier(self, tier: str) -> str:
        tier = (tier or "").strip().lower()
        if tier not in TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Expected one of {', '.join(TIERS)}.")
        return tier

    def set_tier_model(self, tier: str, model: str) -> dict:
        """Assign a model to the low / medium / high slot.

        Changing the slot the agent is currently on switches the live model
        too, so the button does what it looks like it does.
        """
        tier = self._check_tier(tier)
        self.model_tiers[tier] = model.strip()
        if tier == self.active_tier:
            self.change_model(self.model_tiers[tier])
        return self.model_state()

    def select_tier(self, tier: str) -> dict:
        """Flip the live model to whichever one that slot holds."""
        self.active_tier = self._check_tier(tier)
        self.change_model(self.model_tiers[self.active_tier])
        return self.model_state()

    def available_models(self) -> list[str]:
        """The .env catalogue, plus anything the tiers or the live model point
        at that is not in it - a model assigned by hand never disappears from
        the list it was chosen from."""
        models = list(DEFAULT_CLOUD_MODELS)
        for name in [*self.model_tiers.values(), self.model]:
            if name and name not in models:
                models.append(name)
        return models

    def model_state(self) -> dict:
        """Everything the UI needs to draw the model list and the switch."""
        return {
            "model": self.model,
            "tiers": list(TIERS),
            "active_tier": self.active_tier,
            "model_tiers": dict(self.model_tiers),
            "available_models": self.available_models(),
        }
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

    def run_tool(self, name: str, arguments: dict) -> str:
        """Run one tool call and return the text the model should see.

        Failures come back as text rather than raising: the model gets to read
        what went wrong and try again, instead of the whole request dying.
        """
        function = self.functions.get(name)
        if function is None:
            return f"Error: there is no tool named '{name}'."
        try:
            return str(function(**arguments))
        except Exception as e:  # usually the model passing the wrong arguments
            return f"Error: {name} failed: {e}"

    # glm-family thinking models occasionally spend their wrap-up turn on
    # reasoning and return an empty content field - the 2026-09-15 empty
    # replies (POST /chat 200 OK, nothing rendered). What reaches the user has
    # to be a message the model actually addressed to them, so recovery keeps
    # asking - alternating between nudging it in context and having it restate
    # its own reasoning - and the raw reasoning is never shipped as the reply.
    EMPTY_REPLY_NUDGE = (
        "Your previous response arrived with no message content - everything "
        "went into hidden reasoning. Reply now with your final answer to the "
        "user as plain message content: no tool calls, no hidden reasoning."
    )

    # Used when the model has reasoning but keeps failing to speak. It reads
    # its own notes back and writes the message it never sent.
    REASONING_TO_REPLY = (
        "The text below is your own private reasoning towards an answer you "
        "did not manage to send. Write the message the user should receive: "
        "address them directly, in your own voice, giving the answer your "
        "reasoning reached. Do not narrate your thinking, do not mention this "
        "instruction, and do not repeat the notes back. Reply with that "
        "message and nothing else."
    )

    async def _plain_chat(self, messages) -> tuple:
        """One tool-free model call. Returns (content, thinking), never raises.

        think=False asks the server to stop routing the answer into the
        thinking channel; backends that reject the flag get a plain call.
        """
        for extra in ({"think": False}, {}):
            try:
                response = await self.client.chat(
                    model=self.model, messages=messages, **extra
                )
            except Exception as e:
                print(f"[agent] Recovery call failed ({e}).")
                continue
            message = response.message
            return message.content or "", getattr(message, "thinking", None) or ""
        return "", ""

    def _nudge_messages(self, thinking: str) -> list:
        """The conversation so far, with the silent turn shown back to it.

        The model's own empty turn goes in carrying its reasoning, so the
        nudge refers to something the model can actually see - without it the
        transcript ends in two user messages discussing a response that is
        nowhere in the history.
        """
        messages = self.str_mem.get_memory()
        if thinking.strip():
            messages = messages + [{"role": "assistant", "content": thinking.strip()}]
        return messages + [{"role": "user", "content": self.EMPTY_REPLY_NUDGE}]

    def _restate_messages(self, thinking: str) -> list:
        """Just the reasoning and the instruction to turn it into a reply.

        No history and no tools: there is nothing here for the model to get
        lost in, which is why this is the pass that usually lands.
        """
        return [
            {"role": "system", "content": self.REASONING_TO_REPLY},
            {"role": "user", "content": thinking.strip()},
        ]

    async def _recover_reply(self, thinking: str) -> str:
        """Ask again until the model produces a message meant for the user.

        Returns "" if every attempt came back as reasoning-only - the caller
        decides what to say then. Reasoning is never returned as the reply: a
        wall of the model's private notes is not an answer, and shipping it
        reads worse than admitting the turn failed.
        """
        for attempt in range(1, EMPTY_REPLY_RETRIES + 1):
            # Alternate the two approaches. Repeating one failing strategy
            # tends to fail the same way each time.
            restate = bool(thinking.strip()) and attempt % 2 == 0
            messages = (
                self._restate_messages(thinking) if restate
                else self._nudge_messages(thinking)
            )
            how = "restated reasoning" if restate else "in-context nudge"

            content, new_thinking = await self._plain_chat(messages)
            if content.strip() and content.strip() != thinking.strip():
                print(f"[agent] Recovered on attempt {attempt} ({how}).")
                return content

            # The retry may have buried its answer in the thinking channel as
            # well - that is the text the next restate pass should work from.
            if new_thinking.strip():
                thinking = new_thinking
            print(f"[agent] Attempt {attempt} ({how}) came back without message content.")

        return ""

    async def _finalize_reply(self, message, narration: str = "") -> str:
        """Turn the model's final (tool-free) turn into the text the user sees.

        `narration` is any content the model produced earlier in this same
        turn, alongside its tool calls - it was written for the user and is
        about the question actually being asked, so it beats an apology.
        """
        content = message.content or ""
        if content.strip():
            return content

        thinking = getattr(message, "thinking", None) or ""
        print(
            f"[agent] Empty model reply - content empty, thinking={len(thinking)} chars. "
            f"Head: {thinking[:300]!r}"
        )

        recovered = await self._recover_reply(thinking)
        if recovered.strip():
            return recovered

        if narration.strip():
            print("[agent] Recovery exhausted - falling back to this turn's narration.")
            return narration.strip()

        print("[agent] Recovery exhausted with nothing usable to say.")
        return (
            "(My model answered with reasoning instead of a message "
            f"{EMPTY_REPLY_RETRIES} times running, so there is nothing here "
            "worth showing you. Please send that again.)"
        )

    async def chat(self, user_message: str):  # thinking twin
        """Answer the user, letting the model use the workspace tools as it goes."""
        async with self._turn_lock:
            return await self._chat_turn(user_message)

    async def _chat_turn(self, user_message: str) -> str:
        timestamp = str(datetime.now())
        context = self.lt_mem.retrieve_memory(user_message)
        if context:
            user_message += f"\n\nContext from long-term memory:\n{context}"
        await self.str_mem.add_to_memory(
            {"role": "user", "content": f"[{timestamp}] {user_message}"}
        )

        # Content the model emits alongside its tool calls. Models narrate
        # there ("Let me check the workspace first..."), and some put the whole
        # answer there and then fall silent on the wrap-up turn, so it is kept
        # instead of discarded.
        narration = ""

        for _ in range(self.max_iterations):
            response = await self.client.chat(
                model=self.model,
                messages=self.str_mem.get_memory(),
                tools=self.schemas,
            )
            message = response.message

            if (message.content or "").strip():
                narration = message.content

            # No tool calls means this is the actual reply to the user.
            if not message.tool_calls:
                reply = await self._finalize_reply(message, narration)
                # Store what the user actually received instead of the raw
                # model turn, so an empty wrap-up turn never reaches the
                # context window and the record matches the chat log.
                await self.str_mem.add_to_memory({"role": "assistant", "content": reply})
                return reply

            # The model's turn is remembered so that on the next pass it can
            # see the tool calls it just asked for.
            await self.str_mem.add_to_memory(message.model_dump(exclude_none=True))

            for call in message.tool_calls:
                # Tools block on shells, pty waits and file IO - run them in a
                # worker thread so /chat and /health keep answering while a
                # long command runs, instead of stalling the event loop.
                content = await asyncio.to_thread(
                    self.run_tool, call.function.name, call.function.arguments
                )
                await self.str_mem.add_to_memory(
                    {
                        "role": "tool",
                        "tool_name": call.function.name,
                        "content": content,
                    }
                )

        stopped = (
            f"Stopped after {self.max_iterations} rounds of tool calls without "
            f"reaching an answer."
        )
        # Whatever the model managed to say on the way is worth more to the
        # user than the notice on its own.
        reply = f"{narration.strip()}\n\n({stopped})" if narration.strip() else stopped
        await self.str_mem.add_to_memory({"role": "assistant", "content": reply})
        return reply
