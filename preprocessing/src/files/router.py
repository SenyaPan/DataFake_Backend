import shutil
import os
import time
from datetime import datetime

from fastapi import APIRouter, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Union

from preprocessing.src.files.functions_preprocess import check_file, get_hash_md5, check_hash, \
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
async def analyze_file(uploaded_file: UploadFile, model_num: Union[int, None] = None, return_path: bool = False):
    """
    Uploads file for future analysis, checks its extension and media type, sends it to the microservice with
    neuron-network inference
    """
    start_time = time.time()
    file_path = f'received_{str(datetime.now().strftime("%y%m%d_%H%M%S"))}.{uploaded_file.filename.split(".")[-1]}'
    destination = 'preprocessing/media/' + file_path
    if not os.path.isdir('preprocessing/media/'):
        os.mkdir('preprocessing/media/')
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
    finally:
        uploaded_file.file.close()

    file_hash = get_hash_md5(destination)
    if check_hash(file_hash):  # TODO check if hash is already in database
        results = get_result(file_hash)
        return results

    if check_file(destination) == "image":
        result = await analyse_photo(destination, model_num)
        if not result:
            result = {'message': 'File analyzed successfully, but haven`t found any faces', 'response': {}}
            json_response = JSONResponse(status_code=250, content=result)
        else:
            if return_path:
                result['path'] = file_path
            json_response = JSONResponse(status_code=200, content=result)
    elif check_file(destination) == "audio":
        result = analyse_audio(destination)  # idk what parameters there should be
        json_response = JSONResponse(content=result)
    elif check_file(destination) == "video":
        result = await analyse_video(destination, model_num)
        if not result:
            result = {'message': 'File analyzed successfully, but haven`t found any faces', 'response': {}}  # , 'dir_path': ''}
            json_response = JSONResponse(status_code=250, content=result)
        else:
            if return_path:
                result['path'] = file_path
            json_response = JSONResponse(status_code=200, content=result)
        # for key in result['response'].keys():
        #     plt.clf()
        #     plt.plot(result['response'][str(key)])
        #     plt.savefig(''.join(destination.split('.')[:-1]) + '/' + str(key) + '.png')
    else:
        result = {'data': {'message': 'Error! Wrong file type!', 'response': {}}}  # , 'dir_path': ''}}  #
        json_response = JSONResponse(status_code=415, content=result)

    # os.remove(destination)
    end_time = time.time()
    print(end_time - start_time)

    return json_response


@router.get('/{dir_name}/{file_name}')
async def get_file(dir_name: str, file_name: str):
    return FileResponse(f'preprocessing/media/{dir_name}/{file_name}')
