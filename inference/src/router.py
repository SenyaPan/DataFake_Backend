from io import BytesIO

import torch

from fastapi import APIRouter, UploadFile
from typing import Union
from pydantic import BaseModel
from typing import List
from PIL import Image

from inference.photo_video.deepfake_model.model import PhotoInference


router = APIRouter(
    prefix="/inference",
    tags=["Inference"]
)


# class Probability(BaseModel):
#     fake_prob: float
#     real_prob: float
#
#
# class IDs(BaseModel):
#     id: Probability
#
#
# class DataInput(BaseModel):
#     data: List[str]


@router.post("/photo", summary="Analyze faces", response_description="JSON with results of every "
                                                                                         "face analysis")
async def analyse_photo(uploaded_file: UploadFile, model_num: Union[int, None] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_num == 1:
        model_path = 'inference/photo_video/deepfake_model/model_epoch_76.pth'
    elif model_num == 2:
        model_path = 'inference/photo_video/deepfake_model/wild_epoch_21.pth'
    elif model_num == 3:
        model_path = 'inference/photo_video/deepfake_model/model_v3_15.pt'
    elif model_num == 4:
        model_path = 'inference/photo_video/deepfake_model/resnet_detfake_v1.1_1.pt'
    elif model_num == 5:
        model_path = 'inference/photo_video/deepfake_model/resnet_detfake_v1.2_5.pt'
    else:
        model_path = 'inference/photo_video/deepfake_model/resnet_detfake_v1.1_1.pt'
    inference = PhotoInference(model_path, model_num if model_num else 4, device)

    fake_prob = inference.process_photo(uploaded_file)

    return fake_prob


@router.post("/audio")
async def analyse_photo(data: dict):  # maybe we need just path
    audio_path = data["data"]

    return {"status": 200, "data": "Sorry, we do not process audio at the moment."}
