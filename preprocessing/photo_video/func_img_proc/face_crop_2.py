import cv2
import numpy as np

"""
Данный класс вырезает все лица с кадра

return faces - тип данных список всех лиц, соответственно сохранение идет по списку
его можно вызвать, чтоб пройтись по каждому лицу и классифицировать


"""


class FaceExtractor:
    def __init__(self, net):
        self.net = net

    def detect(self, image):
        output = self.net(image)

        boxes = output[0].boxes.xyxy.cpu().numpy()

        return boxes  # return the face in RGB format

    def process_file(self, file_path):
        if isinstance(file_path, str):
            image = cv2.imread(file_path)
        elif isinstance(file_path, np.ndarray):
            image = file_path
        faces = self.detect(image)
        # if faces:
        return faces
