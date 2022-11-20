from os import getenv

ALLOWED_ORIGINS = [getenv('ALLOWED_ORIGINS')]
ALLOWED_METHODS = ['GET']
ALLOWED_HEADERS = ['*']
ALLOW_CREDENTIALS = False
DATABASE_URL = getenv('DATABASE_URL')
