from bson import ObjectId
from pymongo.collection import Collection


class ReviewDatabaseHandler:
    @staticmethod
    def get_user_reviews(
            review_collection: Collection,
            user_id: ObjectId
    ) -> list:
        return list(review_collection.find({'user_id': user_id}))

    @staticmethod
    def get_reviews(
            review_collection: Collection,
    ) -> list:
        return list(review_collection.find({}))
