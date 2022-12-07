from spacy import load
from bson import ObjectId
from datetime import datetime
from pymongo import InsertOne
from gridfs import GridFS, NoFile
from scipy.sparse import coo_matrix
from pymongo.database import Database
from gensim import corpora, similarities

from src.recommender.base_recommender import BaseRecommender
from src.infrastructure.db.db_config import get_grid_fs, get_db
from src.infrastructure.db.sim.sim_database_handler import SimDatabaseHandler
from src.infrastructure.db.review.review_database_handler import ReviewDatabaseHandler
from src.infrastructure.db.alcohol.alcohol_database_handler import AlcoholDatabaseHandler


class SimilarityRecommender(BaseRecommender):
    ALCOHOL_COLUMNS = ['_id', 'name', 'kind', 'type', 'description', 'taste', 'aroma', 'finish', 'country']
    TYPE = 'SIM'

    def __init__(self, grid_fs: GridFS, db: Database):
        try:
            self.__dict__ = self.load(grid_fs.get(self.TYPE))
        except NoFile:
            self.nlp = load('pl_core_news_md')
            self.similarity_matrix = None
            self.fit(db)
            self.save(grid_fs)

    def fit(self, db: Database):
        print(f'[{datetime.now()}]Starting to fit the similarity model.')
        alcohols = AlcoholDatabaseHandler.get_alcohols(db.alcohols, self.ALCOHOL_COLUMNS)

        alcohols_data = [
            f'{alcohol["name"]}, {alcohol["kind"]}, {alcohol["type"]}, ' \
            f'{alcohol["description"]} {", ".join(alcohol["taste"])} {", ".join(alcohol["aroma"])} ' \
            f'{", ".join(alcohol["finish"])}, {alcohol["country"]}'
            for alcohol in alcohols
        ]
        print(f'[{datetime.now()}]Created alcohols data. Beginning processing.')

        docs = self.nlp.pipe(alcohols_data)
        documents = []
        for doc in docs:
            documents.append([token.lemma_.lower() for token in doc if (not token.is_stop and not token.is_punct)])

        print(f'[{datetime.now()}]Processing ended.')
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(document) for document in documents]
        self.similarity_matrix = similarities.MatrixSimilarity(corpus)

        print(f'[{datetime.now()}]Saving similarities to database.')
        self.save_similarities_to_db(db, alcohols)

    @staticmethod
    def recommend(user_id: ObjectId, db: Database, n: int, already_recommended: list[str]) -> list[str]:
        user_reviews = ReviewDatabaseHandler.get_user_reviews(db.reviews, user_id)
        if len(user_reviews) == 0:
            return []

        alcohol_ids = {str(review['alcohol_id']): review['rating'] for review in user_reviews}
        alcohol_ids_keys = list(alcohol_ids.keys())

        user_mean = sum(alcohol_ids.values()) / len(alcohol_ids)

        # 400 most similar alcohols to the rated ones excluding the already recommended
        similar = SimDatabaseHandler.get_similar_alcohols(
            db.lda_sim,
            alcohol_ids_keys,
            already_recommended + alcohol_ids_keys
        )
        targets = set(_['target'] for _ in similar)

        recommendations = dict()
        for target in targets:
            # sum of weighted similarities
            weight_sim_sum = 0
            # sum of similarities
            sim_sum = 0

            rated_alcohols = [_ for _ in similar if _['target'] == target]

            if len(rated_alcohols) > 0:
                # rated alcohols similar to current potential recommendation
                for similar_alcohol in rated_alcohols:
                    # similarity weight calculated as rated alcohol rating minus user mean rating
                    weight = alcohol_ids[similar_alcohol['source']] - user_mean
                    # add alcohol similarity times weight
                    weight_sim_sum += similar_alcohol['sim'] * weight
                    # add plain similarity
                    sim_sum += similar_alcohol['sim']

                recommendations[target] = {
                    # user mean plus sum of weighted similarities divided by sum of plain similarities
                    'prediction': float(user_mean + weight_sim_sum / sim_sum),
                    'sim_alcohols': [_['source'] for _ in rated_alcohols]
                }

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: -float(x[1]['prediction']))[:n]
        return [sorted_recommendation[0] for sorted_recommendation in sorted_recommendations]

    @staticmethod
    def get_similar_alcohols(alcohol_id: ObjectId, db: Database, n: int) -> list[str]:
        return [
            str(alcohol['target']) for alcohol in
            SimDatabaseHandler.get_similar_alcohols_to(
                db.lda_sim,
                str(alcohol_id),
                n
            )
        ]

    def save_similarities_to_db(self, db: Database, alcohols):
        coo = coo_matrix(self.similarity_matrix)
        csr = coo.tocsr()
        xs, ys = coo.nonzero()

        operations = []
        for x, y in zip(xs, ys):
            if x == y:
                continue

            sim = float(csr[x, y])
            if sim <= 0.2:
                continue
            x_id = str(alcohols[x]['_id'])
            y_id = str(alcohols[y]['_id'])
            operations.append(InsertOne({'source': x_id, 'target': y_id, 'sim': sim}))

        SimDatabaseHandler.save_to_db(db.lda_sim, operations)


similarity_recommender = SimilarityRecommender(get_grid_fs(), get_db())
