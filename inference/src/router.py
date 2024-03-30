import torch

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

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
async def analyse_photo(data: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'inference/photo_video/deepfake_model/wild_epoch_36.pth'

    inference = PhotoInference(model_path, device)

    img_paths = data["data"]
    result = {}
    for i, img_path in enumerate(img_paths):
        # prediction_index, predict_name, fake_probability = inference.process_photo(img)
        # result[i] = {'probability': prediction_index, "fake_or_not": predict_name, "fake_probability": fake_probability}
        fake_prob, real_prob = inference.process_photo(img_path)
        result[i] = fake_prob

    return result


@router.post("/audio")
async def analyse_photo(file):  # maybe we need just path
    return {"status": 200, "data": "Sorry, we do not process audio at the moment."}
