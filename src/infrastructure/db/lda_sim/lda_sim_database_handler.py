from pymongo import DESCENDING
from pymongo.collection import Collection


class LdaSimDatabaseHandler:
    @staticmethod
    def save_to_db(collection: Collection, operations: list):
        collection.bulk_write(operations)

    @staticmethod
    def check_if_empty(collection: Collection):
        return collection.estimated_document_count() > 0

    @staticmethod
    def empty_collection(collection: Collection):
        collection.delete_many({})

    @staticmethod
    def get_similar_alcohols(collection: Collection, alcohol_ids: list[str]):
        return list(
            collection.find(
                {
                    'source': {'$in': alcohol_ids},
                    'sim': {'$gt': 0.2}
                }
            )
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
