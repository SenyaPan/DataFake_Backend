import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
# from inference.photo_video.deepfake_model.model_arc_2 import CustomConvNet
from inference.photo_video.deepfake_model.model_arc_1 import FakeCatcher


class PhotoInference:
    def __init__(self, model_path, device):
        self.device = device
        self.model = FakeCatcher(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.to(device)
        self.class_names = ['Fake', 'Real']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def process_photo(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            probabilities = F.softmax(output, dim=1)
            fake_prob = probabilities[0, self.class_names.index('Fake')].item()
            real_prob = probabilities[0, self.class_names.index('Real')].item()

        return fake_prob, real_prob


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

