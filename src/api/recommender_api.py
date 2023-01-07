from gridfs import GridFS
from datetime import datetime
from pymongo.database import Database
from fastapi import APIRouter, status, Depends, Header, HTTPException

from src.recommender.svd_recommender import SVD_recommender
from src.infrastructure.db.db_config import get_db, get_grid_fs
from src.recommender.random_recommender import random_recommender
from src.infrastructure.db.util.object_id_util import validate_object_id
from src.recommender.similarity_recommender import similarity_recommender
from src.infrastructure.db.sim.sim_database_handler import SimDatabaseHandler


router = APIRouter(prefix='/recommend', tags=['recommender'])

RECOMMENDATIONS_NUM = 10


@router.get(
    path='/fit',
    status_code=status.HTTP_204_NO_CONTENT,
    summary='Used to fit the models'
)
async def fit_the_models(
        db: Database = Depends(get_db),
        grid_fs: GridFS = Depends(get_grid_fs),
        x_appengine_cron: bool = Header()
):
    if not x_appengine_cron:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    print(f'[{datetime.now()}]Removing SIM from database.')
    grid_fs.delete(similarity_recommender.TYPE)
    print(f'[{datetime.now()}]Clearing sim database.')
    SimDatabaseHandler.empty_collection(db.lda_sim)
    print(f'[{datetime.now()}]Dropped sim collection from database.')
    SimDatabaseHandler.init_collection(db)
    print(f'[{datetime.now()}]Initialized empty sim collection.')
    similarity_recommender.fit(db)
    similarity_recommender.save(grid_fs)

    print(f'[{datetime.now()}]Removing SVD from database.')
    grid_fs.delete(SVD_recommender.TYPE)
    SVD_recommender.fit(db)
    SVD_recommender.save(grid_fs)


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
    sim_recommendations = similarity_recommender.recommend(
        user_id=user_id,
        db=db,
        n=RECOMMENDATIONS_NUM-len(svd_recommendations)-2,
        already_recommended=svd_recommendations
    )
    random_recommendation = random_recommender.recommend(
        n=RECOMMENDATIONS_NUM-len(sim_recommendations)-len(svd_recommendations),
        user_id=user_id,
        db=db,
        already_recommended=sim_recommendations + svd_recommendations
    )
    return {
        'recommendations': svd_recommendations + sim_recommendations + random_recommendation
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
        'similar': similarity_recommender.get_similar_alcohols(alcohol_id, db, 5)
    }
