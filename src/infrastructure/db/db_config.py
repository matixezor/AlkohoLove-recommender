from gridfs import GridFS
from pymongo import MongoClient
from pymongo.database import Database

from src.config import DATABASE_URL

client = MongoClient(DATABASE_URL)
db: Database = client.alkoholove
grid_fs: GridFS = GridFS(db)


def get_db() -> Database:
    return db


def get_grid_fs() -> GridFS:
    return grid_fs
