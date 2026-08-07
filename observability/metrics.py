class Metrics:


    def __init__(self):

        self.values = {}



    def add(
        self,
        name,
        value
    ):

        self.values[name] = value



    def export(self):

        return self.values
