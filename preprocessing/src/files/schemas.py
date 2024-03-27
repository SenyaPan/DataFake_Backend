from pydantic import BaseModel


class ResultCreate(BaseModel):
    id: int
    hash: str
    id_file: int
    result: dict  # json or array of float?


class FileCreate(BaseModel):
    id: int
    file_path: str
