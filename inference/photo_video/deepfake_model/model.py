from io import BytesIO
from base64 import b64decode as dec64


import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

from inference.photo_video.deepfake_model.model_arc_1 import FakeCatcher1
from inference.photo_video.deepfake_model.model_arc_2 import CustomConvNet
from inference.photo_video.deepfake_model.model_arc_3 import FakeCatcher3
from inference.photo_video.deepfake_model.model_arc_4 import ResNet18

from inference.photo_video.deepfake_model.model_arc_5 import ParallelResNet
from inference.photo_video.deepfake_model.model_arc_5 import block


class PhotoInference:
    def __init__(self, model_path, model_arc, device):
        self.device = device
        if model_arc == 1:
            self.model = FakeCatcher1(device)
        elif model_arc == 2:
            self.model = CustomConvNet(device)
        elif model_arc == 3:
            self.model = FakeCatcher3(device)
        elif model_arc == 4:
            self.model = ResNet18(arr=[2, 2, 2, 2])
            self.model.to(device)
        elif model_arc == 5:
            self.model = ResNet18(arr=[2, 3, 2, 1])
            self.model.to(device)
        elif model_arc == 6:
            self.model = ParallelResNet(block=block, layers=[2, 1, 1, 1], num_classes=2, device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.to(device)
        self.class_names = ['Fake', 'Real']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def process_photo(self, img):
        img = Image.open(img.file)

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)

            probabilities = F.softmax(output, dim=1)
            fake_prob = probabilities[0, self.class_names.index('Fake')].item()
            real_prob = probabilities[0, self.class_names.index('Real')].item()

        return fake_prob #round(fake_prob*100, 1)


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = 'model_epoch_76.pth'
#     face_to_process = '' #сюда кидаем ебало
#
#     inference = PhotoInference(model_path, device)
#
# #   МОЖНО ЗАБИРАТЬ ТОЛЬКО prediction_index
#     prediction_index, predict_name = inference.process_photo(face_to_process)
#     print(prediction_index, predict_name)


#для тестов
'''
for filename in tqdm(os.listdir(photo_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(photo_dir, filename)
        fp_processing(model, img_path, device)


for filename in tqdm(os.listdir(photo_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(photo_dir, filename)
        prediction_index, predict_name = inference.process_photo(file_path)
        print(prediction_index, predict_name)
'''

