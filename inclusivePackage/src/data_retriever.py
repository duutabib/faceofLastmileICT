
class DataRetriever:
    def __init__(self, source= [Enum], is_connected):
        self.source = source
        self.is_connected = is_connected

    def connector(self):
        if not self.is_connected:
            'establish connection'
        return self.is_connected

    def get_data(self):
        # data from source
        if connector():
            'source data and write to csv or EXcel'
            'More details to implement...'
        return 0 
