from pymongo import DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database


class SimDatabaseHandler:
    @staticmethod
    def save_to_db(collection: Collection, operations: list):
        collection.bulk_write(operations)

    @staticmethod
    def check_if_empty(collection: Collection):
        return collection.estimated_document_count() > 0

    @staticmethod
    def empty_collection(collection: Collection):
        collection.drop()

    @staticmethod
    def init_collection(db: Database):
        db.create_collection(name='lda_sim')

    @staticmethod
    def get_similar_alcohols(collection: Collection, alcohol_ids: list[str], already_recommended: list[str]):
        return list(
            collection.find(
                {
                    'source': {'$in': alcohol_ids},
                    'target': {'$nin': already_recommended},
                    'sim': {'$gt': 0.2},
                }
            ).sort('sim', DESCENDING).limit(400)
        )

    @staticmethod
    def get_similar_alcohols_to(collection: Collection, alcohol_id: str, n: int):
        return list(
            collection.find(
                {
                    'source': alcohol_id,
                    'sim': {'$gt': 0.2}
                }
            ).sort('sim', DESCENDING).limit(n)
        )
