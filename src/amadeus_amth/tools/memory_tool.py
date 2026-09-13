"""Memory tools — let the model deliberately store and recall long-term memories.

Long-term memory is already injected into the context ahead of every turn (see
Amadeus.chat), but that is a passive, similarity-based lookup against whatever
the user just said. These tools give the model the active half: writing a fact
down because it decided it was worth keeping, and searching for something the
automatic lookup did not surface.

Unlike workspace_tool, the callables here are bound methods on a
LongTermMemory instance rather than module-level functions, so FUNCTIONS is
built by build_functions(lt_mem) instead of being a module constant.
"""

import functools

from ..memory.lt_memory import LongTermMemory


def _tool(fn):
    """Same contract as workspace_tool._tool: the model always gets a string
    back, and an unexpected failure is reported rather than unwinding into the
    tool-call loop."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"Error: {fn.__name__} failed: {e}"

    return wrapper


def build_functions(lt_mem: LongTermMemory) -> dict:
    """Return the FUNCTIONS mapping for a specific LongTermMemory instance.

    The wrappers exist to translate the storage layer's return values — an id
    or None, a list of documents or None — into the plain text the tool-call
    loop feeds back to the model.
    """

    @_tool
    def save_memory(text: str) -> str:
        print(f"[memory] save_memory: {text!r}")
        memory_id = lt_mem.save_memory(text)
        if memory_id is None:
            return "Error: the memory could not be saved (embedding failed)."
        return f"Saved to long-term memory (id {memory_id})."

    @_tool
    def retrieve_memory(query: str, top_k: int = 2) -> str:
        print(f"[memory] retrieve_memory: {query!r} (top_k={top_k})")
        documents = lt_mem.retrieve_memory(query, top_k=top_k)
        if documents is None:
            return "Error: the memory search failed (embedding failed)."
        if not documents:
            return "No relevant memories found."
        return "\n".join(f"- {doc}" for doc in documents)

    return {
        "save_memory": save_memory,
        "retrieve_memory": retrieve_memory,
    }


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Stores a piece of information in long-term memory so it can be "
                "recalled in future conversations. Use it for durable facts, "
                "preferences and decisions worth remembering — not for details "
                "that only matter for the current exchange."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The fact to remember, written as a single "
                            "self-contained sentence that will still make sense "
                            "without the surrounding conversation."
                        ),
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": (
                "Searches long-term memory for entries similar in meaning to a "
                "query and returns the closest matches. Relevant memories are "
                "already injected automatically each turn, so use this only to "
                "look for something that was not surfaced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, in natural language.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many memories to return. Defaults to 2.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]