
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
        self.memory = []
        self.memory.append({"role": "system", "content": sys_prompt})


    async def add_to_memory(self, message: dict):
        if len(self.memory) >= self.max_memory_size:
            await self.condense_memory()
        self.memory.append(message)

    def get_memory(self):
        return self.memory


    async def condense_memory(self):
        # Index 0 (system prompt) and 1 (first message) survive clear_memory,
        # so only what comes after them needs summarising.
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in self.memory[2:]
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

        self.clear_memory()
        if summary:
            self.memory.append(
                {"role": "system", "content": f"Summary of the earlier conversation:\n{summary}"}
            )


    def clear_memory(self):
        del self.memory[2:]  # doesnt delete the system prompt or the first message
