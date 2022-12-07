from bson import ObjectId
from pandas import DataFrame
from datetime import datetime
from gridfs import GridFS, NoFile
from collections import defaultdict
from pymongo.database import Database
from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV

from src.recommender.base_recommender import BaseRecommender
from src.infrastructure.db.db_config import get_grid_fs, get_db
from src.infrastructure.db.review.review_database_handler import ReviewDatabaseHandler


class SVDRecommender(BaseRecommender):
    TYPE = 'SVD'

    def __init__(self, grid_fs: GridFS, db: Database):
        try:
            self.__dict__ = self.load(grid_fs.get(self.TYPE))
        except NoFile:
            self.model = None
            self.predictions = defaultdict(list)
            self.fit(db)
            self.save(grid_fs)

    def fit(self, db: Database):
        print(f'[{datetime.now()}]Starting to fit the SVD model.')
        reader = Reader(rating_scale=(1.0, 5.0))
        reviews_df = DataFrame(ReviewDatabaseHandler.get_reviews(db.reviews))
        reviews_df['user_id'] = reviews_df['user_id'].apply(lambda x: str(x))
        reviews_df['alcohol_id'] = reviews_df['alcohol_id'].apply(lambda x: str(x))

        data = Dataset.load_from_df(reviews_df[['user_id', 'alcohol_id', 'rating']], reader)

        print(f'[{datetime.now()}]Calculating best parameters.')
        param_grid = {
            'n_epochs': [_ for _ in range(5, 21, 5)],
            'lr_all': [0.002, 0.005, 0.008, 0.010],
            'reg_all': [0.4, 0.6, 0.8]
        }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)
        print(f'[{datetime.now()}]Best params found: {gs.best_params["rmse"]} with score: {gs.best_score}.')

        print(f'[{datetime.now()}]Starting to fit the model.')
        data = data.build_full_trainset()
        self.model = gs.best_estimator['rmse']
        self.model.fit(data)

        print(f'[{datetime.now()}]Making predictions.')
        test_set = data.build_anti_testset()
        predictions = self.model.test(test_set)
        self.predictions = defaultdict(list)

        print(f'[{datetime.now()}]Mapping predictions to each user.')
        for user_id, alcohol_id, _, prediction, __ in predictions:
            self.predictions[user_id].append((alcohol_id, prediction))

        print(f'[{datetime.now()}]Sorting the predictions for each user.')
        for user_id, user_ratings in self.predictions.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            self.predictions[user_id] = user_ratings[:10]

    def recommend(self, user_id: ObjectId, n: int) -> list[str]:
        return [
                   prediction[0] for prediction in self.predictions[str(user_id)]
               ][:n]


SVD_recommender = SVDRecommender(get_grid_fs(), get_db())
