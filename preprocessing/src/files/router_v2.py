import os
import shutil
import time
from datetime import datetime
from typing import Union

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from preprocessing.src.config import pre_logger
from preprocessing.src.files.functions_inference import analyse_photo, analyse_video, analyse_audio
from preprocessing.src.files.functions_preprocess import check_file

router = APIRouter(
    prefix="/v2",
)


@router.post("/analyze/image")
async def analyze_image(uploaded_file: UploadFile, model_num: Union[int, None] = None, return_path: bool = False):
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
        pre_logger.info(f'Image {file_path} has been uploaded successfully.')
    except:
        pre_logger.exception('There was some mistake during image upload.')
    finally:
        uploaded_file.file.close()

    # file_hash = get_hash_md5(destination)
    # if check_hash(file_hash):  # TODO check if hash is already in database
    #     results = get_result(file_hash)
    #     return results

    if check_file(destination) == "image":
        pre_logger.info(f'Uploaded file {file_path} is an image.')
        result = await analyse_photo(destination, model_num)
        if not result:
            result = {'message': 'File analyzed successfully, but haven`t found any faces', 'response': {}}
            json_response = JSONResponse(status_code=250, content=result)
        else:
            if return_path:
                result['path'] = file_path
            json_response = JSONResponse(status_code=200, content=result)
    else:
        pre_logger.info('The format of file is not image.')
        result = {'data': {'message': 'Error! Wrong file type!', 'response': {}}}
        json_response = JSONResponse(status_code=415, content=result)

    # os.remove(destination)
    end_time = time.time()
    pre_logger.info(f'The analysis of image {file_path} took {end_time - start_time} seconds.')

    return json_response


@router.post("/analyze/video")
async def analyze_video(uploaded_file: UploadFile, model_num: Union[int, None] = None, return_path: bool = False):
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
        pre_logger.info(f'Video {file_path} has been uploaded successfully.')
    except:
        pre_logger.exception('There was some mistake during video upload.')
    finally:
        uploaded_file.file.close()

    # file_hash = get_hash_md5(destination)
    # if check_hash(file_hash):  # TODO check if hash is already in database
    #     results = get_result(file_hash)
    #     return results

    # elif check_file(destination) == "audio":
    #     pre_logger.INFO(f'Uploaded file {file_path} is an audio.')
    #     result = analyse_audio(destination)  # idk what parameters there should be
    #     json_response = JSONResponse(content=result)

    if check_file(destination) == "video":
        pre_logger.info(f'Uploaded file {file_path} is a video.')
        result = await analyse_video(destination, model_num)
        if not result['response']:
            result = {'message': 'File analyzed successfully, but haven`t found any faces', 'response': {}}
        if not result:
            result = {'message': 'File analyzed successfully, but haven`t found any faces',
                      'response': {}}
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
        pre_logger.info('The format of file is not video.')
        result = {'data': {'message': 'Error! Wrong file type!', 'response': {}}}
        json_response = JSONResponse(status_code=415, content=result)

    # os.remove(destination)
    end_time = time.time()
    pre_logger.info(f'The analysis of video {file_path} took {end_time - start_time} seconds.')

    return json_response


@router.post("/analyze/audio")
async def analyze_audio(uploaded_file: UploadFile, model_num: Union[int, None] = None, return_path: bool = False):
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
        pre_logger.info(f'Audio {file_path} has been uploaded successfully.')
    except:
        pre_logger.exception('There was some mistake during audio upload.')
    finally:
        uploaded_file.file.close()

    # file_hash = get_hash_md5(destination)
    # if check_hash(file_hash):  # TODO check if hash is already in database
    #     results = get_result(file_hash)
    #     return results

    if check_file(destination) == "audio":
        pre_logger.info(f'Uploaded file {file_path} is an audio.')
        result = await analyse_audio(destination)  # idk what parameters there should be
        response = {'message': 'File analyzed successfully!', 'response': result}
        json_response = JSONResponse(content=response)
    else:
        pre_logger.info('The format of file is not audio.')
        result = {'data': {'message': 'Error! Wrong file type!', 'response': {}}}
        json_response = JSONResponse(status_code=415, content=result)

    # os.remove(destination)
    end_time = time.time()
    pre_logger.info(f'The analysis of audio {file_path} took {end_time - start_time} seconds.')

    return json_response
