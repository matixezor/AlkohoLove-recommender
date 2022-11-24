from os import path, remove
from datetime import datetime
from pymongo.database import Database
from fastapi import APIRouter, status, Depends

from src.infrastructure.db.db_config import get_db
from src.recommender.svd_recommender import SVD_recommender
from src.recommender.lda_recommender import LDA_recommender
from src.recommender.random_recommender import random_recommender
from src.infrastructure.util.object_id_util import validate_object_id
from src.infrastructure.db.lda_sim.lda_sim_database_handler import LdaSimDatabaseHandler


router = APIRouter(prefix='/recommend', tags=['recommender'])

RECOMMENDATIONS_NUM = 10


@router.get(
    path='/fit',
    status_code=status.HTTP_204_NO_CONTENT,
    summary='Used to fit the models'
)
async def fit_the_models(
        db: Database = Depends(get_db)
):
    if path.isfile(LDA_recommender.FILENAME):
        print(f'[{datetime.now()}]Removing LDA pickle.')
        remove(LDA_recommender.FILENAME)
    print(f'[{datetime.now()}]Clearing LDA sim database.')
    LdaSimDatabaseHandler.empty_collection(db.lda_sim)
    LDA_recommender.fit(db)
    LDA_recommender.save()

    if path.isfile(SVD_recommender.FILENAME):
        print(f'[{datetime.now()}]Removing SVD pickle.')
        remove(SVD_recommender.FILENAME)
    SVD_recommender.fit(db)
    SVD_recommender.save()


@router.get(
    path='/users/{user_id}',
    status_code=status.HTTP_200_OK,
    summary='Recommendations for given user'
)
async def get_recommendations_for_user(
        user_id: str,
        db: Database = Depends(get_db)
):
    user_id = validate_object_id(user_id)
    svd_recommendations = SVD_recommender.recommend(user_id=user_id, n=4)
    lda_recommendations = LDA_recommender.recommend(
        user_id=user_id,
        db=db,
        n=RECOMMENDATIONS_NUM-len(svd_recommendations),
        already_recommended=svd_recommendations
    )
    random_recommendation = random_recommender.recommend(
        n=RECOMMENDATIONS_NUM-len(lda_recommendations)-len(svd_recommendations),
        user_id=user_id,
        db=db,
        already_recommended=lda_recommendations + svd_recommendations
    )
    return {
        'recommendations': lda_recommendations + svd_recommendations + random_recommendation
    }


@router.get(
    path='/alcohols/{alcohol_id}',
    status_code=status.HTTP_200_OK,
    summary='Return similar alcohols'
)
async def get_similar_alcohols(
        alcohol_id: str,
        db: Database = Depends(get_db)
):
    alcohol_id = validate_object_id(alcohol_id)
    return {
        'similar': LDA_recommender.get_similar_alcohols(alcohol_id, db, 5)
    }
