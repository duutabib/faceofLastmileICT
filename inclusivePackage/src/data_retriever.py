import Enum 

class Retriever:
    """Class representing DataRetriever"""
    def __init__(self, source: [Enum] = None, is_connected: bool = False):
        self.source = source
        self.is_connected = is_connected

    def connector(self):
        # establish connection
        if not self.is_connected:
            'establish connection'
        return self.is_connected

    def get_data(self):
        # get data from source
        # data from source
        if self.connector():
            'source data and write to csv or EXcel'
            'More details to implement...'
        return 0 
