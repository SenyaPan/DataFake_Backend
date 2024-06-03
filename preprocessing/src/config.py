import torch
from dotenv import load_dotenv
import onnxruntime as ort

import os
import logging

from ultralytics import YOLO

from preprocessing.photo_video.func_img_proc.face_crop_2 import FaceExtractor

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")


pre_logger = logging.getLogger('root')

try:
    model_path = 'preprocessing/photo_video/func_img_proc/model.pt'
    model = YOLO(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    face_extractor = FaceExtractor(model)
    pre_logger.info("Model for face cutting was created successfully.")
except:
    pre_logger.exception("Model for face cutting was created with an error.")


try:
    model_path = 'preprocessing/photo_video/face_vec/vec_model.onnx'

    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        ort_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    else:
        ort_sess = ort.InferenceSession(model_path)
    pre_logger.info("Model for face comparing was created successfully.")
except:
    pre_logger.exception("Model for face comparing was created with an error.")
