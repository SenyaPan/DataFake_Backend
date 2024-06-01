import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

from inference.photo_video.deepfake_model.functions_for_model_arc_6 import noise_proc, fft_proc
from inference.photo_video.deepfake_model.model_arc_4 import ResNet18

from inference.photo_video.deepfake_model.model_arc_5 import ParallelResNet
from inference.photo_video.deepfake_model.model_arc_5 import block

from inference.photo_video.deepfake_model.model_arc_6 import ParallelResNet as ParallelResNet2
from inference.photo_video.deepfake_model.model_arc_6 import Block


class PhotoInference:
    def __init__(self, model_path, model_arc, device):
        self.device = device
        if model_arc == 5:
            self.model = ResNet18(arr=[2, 3, 2, 1])
        elif model_arc == 8:
            self.model = ParallelResNet(block=block, layers=[4, 3, 2, 1], layers_fft=[3, 2, 1, 1], num_classes=2,
                                        device=device)
        elif model_arc == 1:
            self.model = ParallelResNet2(block=Block, layers=[2, 3, 2, 1], layers_fft=[2, 1, 1, 1], num_classes=2,
                                         device=device)
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

        return fake_prob  # round(fake_prob*100, 1)

    def process_photo_without_photo_preprocessing(self, img):
        img = Image.open(img.file)

        noise_an = noise_proc(img).unsqueeze(0).to(self.device)
        fft_an = fft_proc(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(fft_an, noise_an)

            probabilities = F.softmax(output, dim=1)
            fake_prob = probabilities[0, 0].item()
            # real_prob = probabilities[0, 1].item()

        return fake_prob


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


# для тестов
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

