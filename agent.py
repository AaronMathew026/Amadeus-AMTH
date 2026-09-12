import ollama
from memory.str_memory import STR_Memory








class Amadeus:
    def __init__ (self, model = "gemma4:31b-cloud", system_prompt = "You are Amadeus, a personal AI assistant created by Aaron. You are a large language model that can answer questions, provide information, and assist with various tasks. You are knowledgeable in a wide range of topics and can provide helpful and accurate responses. You are also capable of engaging in casual conversation and providing entertainment. Your goal is to be helpful, informative, and engaging for the user."):
        self.system_prompt = system_prompt
        self.model = model
        self.client = ollama.AsyncClient()
        self.str_mem = STR_Memory(self.system_prompt)




    def fetch_STR_memory(self):
        pass




    async def chat(self, user_message: str): # thinking twin
        self.str_mem.add_to_memory({"role": "user", "content": user_message})
        response = await self.client.chat(
            model = self.model,
            messages =  self.str_mem.get_memory()

        )
        reply = response.message.content
        self.str_mem.add_to_memory({"role": "assistant", "content": reply})
        return reply