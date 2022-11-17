from bson.objectid import ObjectId
from fastapi import HTTPException, status


def validate_object_id(object_id: str):
    if ObjectId.is_valid(object_id):
        return ObjectId(object_id)
    else:
        raise InvalidObjectIdException(object_id)


class InvalidObjectIdException(HTTPException):
    def __init__(self, object_id):
        super().__init__(
            status.HTTP_400_BAD_REQUEST,
            f'{object_id} is not a valid ObjectId, it must be a 12-byte input or a 24-character hex string'
        )
