
class DataRetriever:
    def __init__(self, source= [Enum], is_connected):
        self.source = source
        self.is_connected = is_connected

    def connector(self):
        if not self.is_connected:
            'establish connection'
        return self.is_connected

    def get_data(self):
        if connector():
            'source data and write to csv or EXcel'
        return 0 
