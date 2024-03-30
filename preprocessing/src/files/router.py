import shutil
import os
import time

import matplotlib.pyplot as plt

from fastapi import APIRouter, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

# from preprocessing.src.database import get_async_session
# from preprocessing.src.operations.models import operation
# from preprocessing.src.operations.schemas import OperationCreate

from preprocessing.src.files.functions_preprocess import check_extension, get_file_format, get_hash_md5, check_hash, \
    get_result

from preprocessing.src.files.functions_inference import analyse_photo, analyse_audio, analyse_video


router = APIRouter(
    prefix="/files",
    tags=["Files"]
)

templates = Jinja2Templates(directory="preprocessing/src/templates")


class Probability(BaseModel):
    fake_prob: float
    real_prob: float


class IDs(BaseModel):
    id: Probability


class Result(BaseModel):
    message: str
    response: IDs
    dir_path: str

# @router.get("/analyze")
# def get_template(request: Request):
#     return templates.TemplateResponse("upload.html", {"request": request})


@router.post("/analyze", response_model=Result, summary="Upload and analyze file", response_description="JSON with "
                                                                                                        "results of "
                                                                                                        "analysis")
async def analyze_file(uploaded_file: UploadFile) -> JSONResponse:
    """
    Uploads file for future analysis, checks its extension and media type, sends it to the microservice with
    neuron-network inference
    """
    start_time = time.time()
    if check_extension(uploaded_file):  # probably should be done more properly
        destination = f"preprocessing/media/{uploaded_file.filename}"  # create path for temporary save
        try:
            with open(destination, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
        finally:
            uploaded_file.file.close()

        file_hash = get_hash_md5(destination)

        file_format = get_file_format(destination)
        if check_hash(file_hash):  # TODO check if hash is already in database
            results = get_result(file_hash)
            return results
        elif file_format == "photo":
            result = await analyse_photo(destination)
        elif file_format == "audio":
            result = analyse_audio()  # idk what parameters there should be
        elif file_format == "video":
            result = await analyse_video(destination)
            for key in result['response'].keys():
                plt.clf()
                plt.plot(result['response'][str(key)])
                plt.savefig(''.join(destination.split('.')[:-1]) + '/' + str(key) + '.png')

        json_response = JSONResponse(content=result)

        os.remove(destination)
        end_time = time.time()
        print(end_time - start_time)

        return json_response
    else:
        return {"status": 401, "data": "error: Wrong file format!"}
