import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, UploadFile
from typing import Union
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import librosa

from inference.src.config import inference_1, inference_5, inference_8, inf_logger, audio_model, device

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
    try:
        if model_num == 1:
            fake_prob = inference_1.process_photo_without_photo_preprocessing(uploaded_file)
        elif model_num == 5:
            fake_prob = inference_5.process_photo(uploaded_file)
        elif model_num == 8:
            fake_prob = inference_8.process_photo(uploaded_file)
        else:
            fake_prob = inference_5.process_photo(uploaded_file)
        inf_logger.info(f'Analysis of face {uploaded_file.filename} was successful.')
    except:
        inf_logger.exception(f'During the analysis of face {uploaded_file.filename} an error occurred.')

    return fake_prob


@router.post("/audio")
async def analyse_photo(uploaded_file: UploadFile):  # maybe we need just path
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(uploaded_file.file)
    image = np.array(image.convert('RGB'))
    inputs = transform(image)
    inputs = torch.unsqueeze(inputs, 0).to(device)
    with torch.no_grad():
        output = audio_model(inputs)
        probabilities = F.softmax(output, dim=0)
        _, probabilities = torch.max(output, 1)
        return int(probabilities)
