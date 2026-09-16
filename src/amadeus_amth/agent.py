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

    async def _finalize_reply(self, message) -> str:
        """Turn the model's final (tool-free) turn into the text the user sees."""
        content = message.content or ""
        if content.strip():
            return content

        thinking = getattr(message, "thinking", None) or ""
        print(
            f"[agent] Empty model reply — content empty, thinking={len(thinking)} chars. "
            f"Head: {thinking[:300]!r}"
        )
        print("[agent] Retrying once with an explicit answer-as-content nudge.")

        response = await self.client.chat(
            model=self.model,
            messages=self.str_mem.get_memory()
            + [{"role": "user", "content": self.EMPTY_REPLY_NUDGE}],
        )
        retried = response.message.content or ""
        if retried.strip():
            print("[agent] Retry recovered the reply.")
            return retried

        print("[agent] Retry also empty — falling back to last non-empty assistant turn.")
        for past in reversed(self.str_mem.get_memory()):
            if past.get("role") == "assistant" and (past.get("content") or "").strip():
                return past["content"]
        return (
            "(My model returned an empty reply twice and I had nothing usable on "
            "record — please send that again.)"
        )

    async def chat(self, user_message: str):  # thinking twin
        timestamp = str(datetime.now())
        """Answer the user, letting the model use the workspace tools as it goes."""
        context = self.lt_mem.retrieve_memory(user_message)
        if context:
            user_message += f"\n\nContext from long-term memory:\n{context}"
        await self.str_mem.add_to_memory({"role": "user", "content": f"[{timestamp}] {user_message}"})


        for _ in range(self.max_iterations):
            response = await self.client.chat(
                model=self.model,
                messages=self.str_mem.get_memory(),
                tools=self.schemas,
            )
            message = response.message


            # No tool calls means this is the actual reply to the user.
            if not message.tool_calls:
                reply = await self._finalize_reply(message)
                # Store what the user actually received instead of the raw
                # model turn, so an empty wrap-up turn never reaches the
                # context window and the record matches the chat log.
                await self.str_mem.add_to_memory({"role": "assistant", "content": reply})
                return reply

            # The model's turn is remembered so that on the next pass it can
            # see the tool calls it just asked for.
            await self.str_mem.add_to_memory(message.model_dump(exclude_none=True))

            for call in message.tool_calls:
                # Tools block on shells, pty waits and file IO — run them in a
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

        return (
            f"Stopped after {self.max_iterations} rounds of tool calls without "
            f"reaching an answer."
        )
