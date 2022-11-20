from datetime import datetime
from pickle import dump, load, HIGHEST_PROTOCOL


class BaseRecommender:
    FILENAME = None

    def save(self):
        print(f'[{datetime.now()}]Saving self to file.')
        with open(self.FILENAME, 'wb') as file:
            dump(self.__dict__, file, protocol=HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        print(f'[{datetime.now()}]Loading self from file.')
        with open(filename, 'rb') as file:
            return load(file)
