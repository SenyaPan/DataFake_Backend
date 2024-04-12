from io import BytesIO

import torch

from fastapi import APIRouter, UploadFile
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
async def analyse_photo(uploaded_file: UploadFile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'inference/photo_video/deepfake_model/model_epoch_76.pth'

    inference = PhotoInference(model_path, device)

    fake_prob = inference.process_photo(uploaded_file)

    return fake_prob


@router.post("/audio")
async def analyse_photo(data: dict):  # maybe we need just path
    audio_path = data["data"]

    return {"status": 200, "data": "Sorry, we do not process audio at the moment."}
