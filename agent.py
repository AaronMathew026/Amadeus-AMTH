import ollama






class Amadeus:
    def __init__ (self, model = "gemma4:31b-cloud", system_prompt = "You are Amadeus, a personal AI assistant created by Aaron. You are a large language model that can answer questions, provide information, and assist with various tasks. You are knowledgeable in a wide range of topics and can provide helpful and accurate responses. You are also capable of engaging in casual conversation and providing entertainment. Your goal is to be helpful, informative, and engaging for the user."):
        self.model = model
        self.system_prompt = system_prompt
        self.client = ollama.AsyncClient()



    def fetch_STR_memory(self):
        pass



    async def chat(self, user_message: str, STR_memory: dict = {},thinking: bool = False):
        response = await self.client.chat(
            model = self.model,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ], ## Replace with STR memory when made

        )
        reply = response.message.content
        return reply