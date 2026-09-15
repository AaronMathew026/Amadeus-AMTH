
import asyncio

import ollama

CONDENSE_PROMPT = (
    "Condense the following conversation into a single message that captures the "
    "key points and context. Be concise but keep the essential information: facts "
    "about the user, decisions made, and any tasks still in progress."
)


class STR_Memory():

    def __init__(self, sys_prompt, model, client=None):

        self.max_memory_size = 30
        self.model = model
        self.client = client or ollama.AsyncClient()
        # Serializes add_to_memory's check/condense/append: two concurrent
        # chats (a real /chat plus the hourly heartbeat beat) must not both
        # trigger condense_memory, and nothing may append while a condense's
        # Ollama round-trip is in flight.
        # See shared/reports/memory_race_fix_guide.md.
        self._lock = asyncio.Lock()
        self.memory = []
        self.memory.append({"role": "system", "content": sys_prompt})


    async def add_to_memory(self, message: dict):
        async with self._lock:
            if len(self.memory) >= self.max_memory_size:
                await self.condense_memory()
            self.memory.append(message)

    def get_memory(self):
        # Copy, not reference: stops a concurrent append from landing inside
        # another request's context mid-serialization.
        return list(self.memory)


    async def condense_memory(self):
        # Snapshot BEFORE the await: summarize exactly this slice. With
        # add_to_memory holding the lock this equals the old clear_memory()
        # wipe, but the invariant now lives here instead of trusting every
        # caller — and a lock-bypassing late arrival would land AFTER the
        # summary instead of being deleted unseen.
        snapshot = self.memory[1:]
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in snapshot
        )
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            )
            summary = response.message.content
        except Exception as e:
            # Better to lose the old context than to let memory grow forever.
            print(f"Condense memory error: {e}")
            summary = None

        # Delete ONLY the slice that was summarized — never anything
        # appended while the Ollama round-trip was in flight.
        del self.memory[1 : 1 + len(snapshot)]
        if summary:
            self.memory.insert(
                1,
                {"role": "system", "content": f"Summary of the earlier conversation:\n{summary}"},
            )


    def clear_memory(self):
        # Manual-reset utility; condense_memory now slice-deletes instead.
        # Keep only the system prompt. The old version also pinned the first
        # user message, so the conversation opener sat at index 1 forever and
        # was replayed into every context window after each condense cycle —
        # the repeating "ghost" messages of the 2026-09-13 night watch. The
        # first message is summarised with everything else now.
        del self.memory[1:]
