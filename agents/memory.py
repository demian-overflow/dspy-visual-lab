class AgentMemory:


    def __init__(self):

        self.history = []


    def add(
        self,
        state
    ):

        self.history.append(
            state.model_dump()
        )


    def previous(self):

        return self.history
