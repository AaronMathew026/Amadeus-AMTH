import os

import ollama
from dotenv import load_dotenv

from .memory.str_memory import STR_Memory
from .memory.lt_memory import LongTermMemory
from .tools import memory_tool
from .tools.workspace_tool import FUNCTIONS, SCHEMAS
from datetime import datetime, timedelta

load_dotenv()

# One iteration is one model turn: the model either answers — and we are done —
# or it asks for tools, we run them, and it gets another turn with the results.
# The cap is what stops a model that keeps calling tools forever.
MAX_ITERATIONS = 10

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


class Amadeus:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        startup_message: str = "Your main server is running. This may the first time you have been launched. On the next user message check  for the Following files. IDENTITY.md,SOUL.md,USER.md, once the user sends their message, check if these files exist, if they do, continue as if you already know them. If these files are mssing, promptly ask them for the information required to fill the files in. THIS IS A CRITICAL MESSAGE, DO NOT FORGET IT.",
    ):
        self.model = model
        self.client = ollama.AsyncClient()
        self.system_prompt = system_prompt + "\n\n" + startup_message
        self.str_mem = STR_Memory(self.system_prompt)
        self.lt_mem = LongTermMemory()

        # The tool registry. Workspace tools are plain module-level functions,
        # so they can be shared; the memory tools are bound to this agent's own
        # LongTermMemory and have to be built per instance.
        self.schemas = SCHEMAS + memory_tool.SCHEMAS
        self.functions = {**FUNCTIONS, **memory_tool.build_functions(self.lt_mem)}

    def fetch_STR_memory(self):
        pass

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

    async def chat(self, user_message: str):  # thinking twin
        timestamp = str(datetime.now())
        """Answer the user, letting the model use the workspace tools as it goes."""
        context = self.lt_mem.retrieve_memory(user_message)
        if context:
            user_message += f"\n\nContext from long-term memory:\n{context}"
        self.str_mem.add_to_memory({"role": "user", "content": f"[{timestamp}] {user_message}"})


        for _ in range(MAX_ITERATIONS):
            response = await self.client.chat(
                model=self.model,
                messages=self.str_mem.get_memory(),
                tools=self.schemas,
            )
            message = response.message
        

            # The model's turn is remembered either way, so that on the next
            # pass it can see the tool calls it just asked for.
            self.str_mem.add_to_memory(message.model_dump(exclude_none=True))

            # No tool calls means this is the actual reply to the user.
            if not message.tool_calls:
                return message.content

            for call in message.tool_calls:
                self.str_mem.add_to_memory(
                    {
                        "role": "tool",
                        "tool_name": call.function.name,
                        "content": self.run_tool(
                            call.function.name, call.function.arguments
                        ),
                    }
                )

        return (
            f"Stopped after {MAX_ITERATIONS} rounds of tool calls without reaching "
            f"an answer."
        )
