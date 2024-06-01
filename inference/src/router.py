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
        model_path = 'inference/photo_video/deepfake_model/resnet_focal_loss_v2.3_19.pth'
    elif model_num == 5:
        model_path = 'inference/photo_video/deepfake_model/resnet_detfake_v1.2_5.pt'
    elif model_num == 8:
        model_path = 'inference/photo_video/deepfake_model/resnet_focal_loss_v2.2_1.pth'
    else:
        model_path = 'inference/photo_video/deepfake_model/resnet_detfake_v1.2_5.pt'
    inference = PhotoInference(model_path, model_num if model_num else 5, device)

    if model_num == 1:
        fake_prob = inference.process_photo_without_photo_preprocessing(uploaded_file)
    else:
        fake_prob = inference.process_photo(uploaded_file)

    return fake_prob


# @router.post("/audio")
# async def analyse_photo(data: dict):  # maybe we need just path
#     audio_path = data["data"]
#
#     return {"status": 200, "data": "Sorry, we do not process audio at the moment."}
