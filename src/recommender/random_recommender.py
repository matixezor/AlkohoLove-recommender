from bson import ObjectId
from pymongo.database import Database

from src.infrastructure.db.review.review_database_handler import ReviewDatabaseHandler
from src.infrastructure.db.alcohol.alcohol_database_handler import AlcoholDatabaseHandler


class RandomRecommender:
    def __init__(self):
        pass

    def recommend(self, n: int, user_id: str, db: Database, already_recommended: list[str]) -> list[str]:
        user_reviews = [
            review['alcohol_id'] for review in ReviewDatabaseHandler.get_user_reviews(db.reviews, ObjectId(user_id))
        ]
        already_recommended = [ObjectId(recommended_id) for recommended_id in already_recommended]
        return [
            str(alcohol['_id']) for alcohol in
            AlcoholDatabaseHandler.get_random_n_alcohols(db.alcohols, n, user_reviews + already_recommended)
        ]


random_recommender = RandomRecommender()
