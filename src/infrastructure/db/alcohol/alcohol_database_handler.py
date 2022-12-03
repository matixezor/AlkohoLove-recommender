from bson import ObjectId
from pymongo.collection import Collection


class AlcoholDatabaseHandler:
    @staticmethod
    def get_alcohols(collection: Collection, columns: list[str]) -> list[dict]:
        return list(collection.find({}, {field_name: 1 for field_name in columns}))

    @staticmethod
    def count_types(collection: Collection) -> int:
        return len(collection.distinct("type"))

    @staticmethod
    def get_random_n_alcohols(collection: Collection, n: int, to_omit: list[ObjectId]):
        return list(
            collection.aggregate([
                {
                    '$match': {'_id': {'$nin': to_omit}}
                },
                {'$sort': {'avg_rating': -1}},
                {'$limit': 500},
                {'$sample': {'size': n}}
            ])
        )
