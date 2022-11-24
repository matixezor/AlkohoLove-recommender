from os import path
from spacy import load
from bson import ObjectId
from datetime import datetime
from pymongo import InsertOne
from scipy.sparse import coo_matrix
from pymongo.database import Database
from gensim import corpora, models, similarities

from src.infrastructure.db.db_config import get_db
from src.recommender.base_recommender import BaseRecommender
from src.infrastructure.db.review.review_database_handler import ReviewDatabaseHandler
from src.infrastructure.db.lda_sim.lda_sim_database_handler import LdaSimDatabaseHandler
from src.infrastructure.db.alcohol.alcohol_database_handler import AlcoholDatabaseHandler


class LDARecommender(BaseRecommender):
    ALCOHOL_COLUMNS = ['_id', 'name', 'kind', 'type', 'description', 'taste', 'aroma', 'finish', 'country']
    FILENAME = 'LDARecommender.pickle'

    def __init__(self, db: Database):
        if path.isfile(self.FILENAME):
            self.__dict__ = self.load(self.FILENAME)
        else:
            self.nlp = load('pl_core_news_md')
            self.dictionary = None
            self.corpus = None
            self.lda_model = None
            self.index = None
            self.alcohols = None
            self.fit(db)
            self.save()

    def fit(self, db: Database):
        print(f'[{datetime.now()}]Starting to fit the LDA model.')
        num_topics = AlcoholDatabaseHandler.count_types(db.alcohols)
        self.alcohols = AlcoholDatabaseHandler.get_alcohols(db.alcohols, self.ALCOHOL_COLUMNS)

        alcohols_data = [
            f'{alcohol["name"]}, {alcohol["kind"]}, {alcohol["type"]}, ' \
            f'{alcohol["description"]} {", ".join(alcohol["taste"])} {", ".join(alcohol["aroma"])} ' \
            f'{", ".join(alcohol["finish"])}, {alcohol["country"]}'
            for alcohol in self.alcohols
        ]
        print(f'[{datetime.now()}]Created alcohols data. Beginning processing.')

        docs = self.nlp.pipe(alcohols_data)
        documents = []
        for doc in docs:
            documents.append([token.lemma_ for token in doc if (not token.is_stop and not token.is_punct)])

        print(f'[{datetime.now()}]Processing ended.')
        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(document) for document in documents]
        self.lda_model = models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics
        )
        self.index = similarities.MatrixSimilarity(self.corpus)

        print(f'[{datetime.now()}]Saving similarities to database.')
        self.save_similarities_to_db(db)

    @staticmethod
    def recommend(user_id: ObjectId, db: Database, n: int, already_recommended: list[str]) -> list[str]:
        user_reviews = ReviewDatabaseHandler.get_user_reviews(db.reviews, user_id)
        if len(user_reviews) == 0:
            return []

        alcohol_ids = {str(review['alcohol_id']): review['rating'] for review in user_reviews}

        user_mean = sum(alcohol_ids.values()) / len(alcohol_ids)

        similar = LdaSimDatabaseHandler.get_similar_alcohols(
            db.lda_sim,
            list(alcohol_ids.keys()),
            already_recommended
        )
        targets = set(_['target'] for _ in similar)

        recommendations = dict()
        for target in targets:
            pre = 0
            sim_sum = 0

            rated_alcohols = [_ for _ in similar if _['target'] == target]

            if len(rated_alcohols) > 0:
                for similar_alcohol in rated_alcohols:
                    r = alcohol_ids[similar_alcohol['source']] - user_mean
                    pre += similar_alcohol['sim'] * r
                    sim_sum += similar_alcohol['sim']

                if sim_sum > 0:
                    recommendations[target] = {
                        'prediction': float(user_mean + pre / sim_sum),
                        'sim_alcohols': [_['source'] for _ in rated_alcohols]
                    }
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: -float(x[1]['prediction']))[:n]
        return [sorted_recommendation[0] for sorted_recommendation in sorted_recommendations]

    @staticmethod
    def get_similar_alcohols(alcohol_id: ObjectId, db: Database, n: int) -> list[str]:
        return [
            str(alcohol['target']) for alcohol in
            LdaSimDatabaseHandler.get_similar_alcohols_to(
                db.lda_sim,
                str(alcohol_id),
                n
            )
        ]

    def save_similarities_to_db(self, db: Database):
        coo = coo_matrix(self.index)
        csr = coo.tocsr()
        xs, ys = coo.nonzero()

        operations = []
        for x, y in zip(xs, ys):
            if x == y:
                continue

            sim = float(csr[x, y])
            if sim < 0.1:
                continue
            x_id = str(self.alcohols[x]['_id'])
            y_id = str(self.alcohols[y]['_id'])
            operations.append(InsertOne({'source': x_id, 'target': y_id, 'sim': sim}))

        LdaSimDatabaseHandler.save_to_db(db.lda_sim, operations)


LDA_recommender = LDARecommender(get_db())
