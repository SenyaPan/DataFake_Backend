import torch
import logging

from inference.photo_video.deepfake_model.model import PhotoInference

inf_logger = logging.getLogger('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model_path_1 = 'inference/photo_video/deepfake_model/resnet_focal_loss_v2.3_19.pth'
    inference_1 = PhotoInference(model_path_1, 1, device)
    inf_logger.info("Model #1 was created successfully.")
except:
    inf_logger.exception("During the creation of model #1 an error occurred.")

try:
    model_path_5 = 'inference/photo_video/deepfake_model/resnet_detfake_v1.2_5.pt'
    inference_5 = PhotoInference(model_path_5, 5, device)
    inf_logger.info("Model #5 was created successfully.")
except:
    inf_logger.exception("During the creation of model #5 an error occurred.")
model_path_8 = 'inference/photo_video/deepfake_model/resnet_focal_loss_v2.2_1.pth'

try:
    inference_8 = PhotoInference(model_path_8, 8, device)
    inf_logger.info("Model #8 was created successfully.")
except:
    inf_logger.exception("During the creation of model #8 an error occurred.")
