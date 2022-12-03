from datetime import datetime
from gridfs import GridFS, GridOut
from pickle import dumps, load, HIGHEST_PROTOCOL


class BaseRecommender:
    TYPE = None

    def save(self, grid_fs: GridFS):
        print(f'[{datetime.now()}]Saving self to database with id [{self.TYPE}].')
        grid_fs.put(dumps(self.__dict__), _id=self.TYPE, protocol=HIGHEST_PROTOCOL)

    def load(self, file: GridOut):
        print(f'[{datetime.now()}]Loading self [{self.TYPE}] from database.')
        return load(file)
