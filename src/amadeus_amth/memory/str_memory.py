


class STR_Memory():

    def __init__(self,sys_prompt):
        
        self.max_memory_size = 50
        self.memory = []
        self.memory.append({"role": "system", "content": sys_prompt})


    def add_to_memory(self,dict :dict):
        if len(self.memory) >= self.max_memory_size:
            self.condense_memory()
        self.memory.append(dict)

    def get_memory(self):
        return self.memory


    def condense_memory(self):
        # This is a placeholder for the condense memory logic

        self.clear_memory



    def clear_memory(self):
        for i in range(len(self.memory)-1, 1, -1): # doesnt delete the system prompt or the first user message
            del self.memory[i]

    