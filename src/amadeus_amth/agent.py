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
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        startup_message: str = "Your main server is running. This may the first time you have been launched. On the next user message check  for the Following files. IDENTITY.md,SOUL.md,USER.md, once the user sends their message, check if these files exist, if they do, continue as if you already know them. If these files are mssing, promptly ask them for the information required to fill the files in. THIS IS A CRITICAL MESSAGE, DO NOT FORGET IT.",

    ):
        self.model = model
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
    # replies (POST /chat 200 OK, nothing rendered). Recovery ladder: one
    # retry that shows the model its own reasoning back and asks for it as
    # content, then the reasoning itself, then anything the model said
    # earlier in this same turn, then a spoken last resort.
    # Never ship an empty string.
    EMPTY_REPLY_NUDGE = (
        "Your previous response arrived with no message content - everything "
        "went into hidden reasoning. Reply now with your final answer to the "
        "user as plain message content: no tool calls, no hidden reasoning."
    )

    async def _retry_for_content(self, thinking: str) -> str:
        """Ask once more for the answer as message content.

        The model's own empty turn goes back in with its reasoning attached,
        so the nudge refers to something the model can actually see - the
        earlier version sent two user messages in a row and asked about a
        response that was nowhere in the transcript. think=False asks the
        server to stop routing the answer into the thinking channel.
        """
        messages = self.str_mem.get_memory()
        if thinking.strip():
            messages = messages + [{"role": "assistant", "content": thinking.strip()}]
        messages = messages + [{"role": "user", "content": self.EMPTY_REPLY_NUDGE}]

        try:
            response = await self.client.chat(
                model=self.model, messages=messages, think=False
            )
        except Exception as e:
            # think=False is rejected by some backends; try again without it
            # rather than turning a recoverable empty turn into a 500.
            print(f"[agent] Nudge with think=False failed ({e}); retrying plain.")
            try:
                response = await self.client.chat(model=self.model, messages=messages)
            except Exception as e2:
                print(f"[agent] Nudge failed: {e2}")
                return ""
        return response.message.content or ""

    async def _finalize_reply(self, message, narration: str = "") -> str:
        """Turn the model's final (tool-free) turn into the text the user sees.

        `narration` is any content the model produced earlier in this same
        turn, alongside its tool calls - it is about the question actually
        being asked, so it beats an apology.
        """
        content = message.content or ""
        if content.strip():
            return content

        thinking = getattr(message, "thinking", None) or ""
        print(
            f"[agent] Empty model reply - content empty, thinking={len(thinking)} chars. "
            f"Head: {thinking[:300]!r}"
        )
        print("[agent] Retrying once with an explicit answer-as-content nudge.")

        retried = await self._retry_for_content(thinking)
        if retried.strip():
            print("[agent] Retry recovered the reply.")
            return retried

        # For these models the answer usually IS the reasoning, so speaking it
        # beats dropping it. The previous version replayed the last non-empty
        # assistant turn instead, which handed the user a stale answer to a
        # question they had already asked.
        if thinking.strip():
            print("[agent] Retry also empty - answering from the model's reasoning.")
            return thinking.strip()

        if narration.strip():
            print("[agent] Retry also empty - falling back to this turn's narration.")
            return narration.strip()

        print("[agent] Nothing usable in the model's turn.")
        return (
            "(My model finished that turn without saying anything - no content "
            "and no reasoning to fall back on. Please send that again.)"
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
