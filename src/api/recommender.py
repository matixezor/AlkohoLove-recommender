from pymongo.database import Database
from fastapi import APIRouter, status, Depends

from src.infrastructure.db.db_config import get_db
from src.recommender.random_recommender import random_recommender
from src.recommender.svd_recommender import SVD_recommender
from src.recommender.lda_recommender import LDA_recommender
from src.infrastructure.util.object_id_util import validate_object_id

router = APIRouter(prefix='/recommend', tags=['recommender'])


@router.get(
    path='/{user_id}',
    status_code=status.HTTP_200_OK,
    summary='Recommendations for given user'
)
async def get_recommendations(
        user_id: str,
        db: Database = Depends(get_db)
):
    user_id = validate_object_id(user_id)
    lda_recommendations = LDA_recommender.recommend(user_id, db, 4)
    svd_recommendations = SVD_recommender.recommend(user_id, 4, lda_recommendations)
    random_recommendation = random_recommender.recommend(4, user_id, db, lda_recommendations + svd_recommendations)
    return lda_recommendations + svd_recommendations + random_recommendation
