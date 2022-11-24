from uvicorn import run
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.recommender import router as recommender_router
from src.infrastructure.config.config import \
    ALLOWED_ORIGINS, ALLOWED_HEADERS, ALLOWED_METHODS, ALLOW_CREDENTIALS, DOCS_URL

app = FastAPI(title='AlkohoLove-recommender', docs_url=DOCS_URL, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)

app.include_router(recommender_router)


if __name__ == '__main__':
    run('main:app', host='127.0.0.1', port=8081, reload=True)
