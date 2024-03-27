import os
import secrets
import cv2
import numpy as np

"""
Данный класс сохраняет все лица с кадра, вот теперь точно

return faces - тип данных список всех лиц, соответственно сохранение идет по списку
его можно вызвать, чтоб пройтись по каждому лицу и классифицировать


"""


class FaceExtractor:
    def __init__(self, net):
        self.net = net

    def detect(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                face = cv2.cvtColor(image[startY:endY, startX:endX], cv2.COLOR_BGR2RGB)  # convert BGR to RGB
                faces.append(face)

        return faces  # return the face in RGB format

    def save_faces(self, faces, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for idx, face in enumerate(faces):
            try:
                random_name = secrets.token_hex(8)
                image_name = f"{random_name}_{idx}.jpg"
                image_path = os.path.join(directory, image_name)
                cv2.imwrite(image_path, face)
            except Exception as ex:
                print(f"Error saving face {idx}: {ex}")

    def process_file(self, file_path):
        # print("Обработка изображения:", file_path)
        if isinstance(file_path, str):
            image = cv2.imread(file_path)
        elif isinstance(file_path, np.ndarray):
            image = file_path
        faces = self.detect(image)
        # if faces:
        return faces
        # elif file_path.endswith(('.mp4', '.avi', '.mov')):  # Обработка видео
        #     # print("Обработка видео:", file_path)
        #     vidcap = cv2.VideoCapture(file_path)
        #     while True:
        #         ret, frame = vidcap.read()
        #         if not ret:
        #             break
        #         faces = self.detect(frame)
        #         if faces:
        #             self.save_faces(faces, out)
        #             return faces
        # else:
        #     print("Неподдерживаемый формат файла")


"""
Данный класс сохраняет все лица с кадра, вот теперь точно

yield face - тип данных !numpy.ndarray!


"""
