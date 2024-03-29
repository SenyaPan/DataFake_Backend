import cv2
import os
import requests
import zipfile
import onnxruntime as ort

from moviepy.editor import *

from PIL import Image
from fastapi.responses import JSONResponse, FileResponse

from preprocessing.photo_video.func_img_proc.face_crop import FaceExtractor
from preprocessing.photo_video.face_vec.feature_vec import img2arr, cos_vec


def cut_faces(filename, frame=None):
    net = cv2.dnn.readNetFromCaffe('preprocessing/photo_video/func_img_proc/deploy.prototxt',
                                   'preprocessing/photo_video/func_img_proc/weights.caffemodel')
    face_extractor = FaceExtractor(net)

    if frame is None:
        face_frames = face_extractor.process_file(filename)
    else:
        face_frames = face_extractor.process_file(frame)

    dir_for_save = ''.join(filename.split('/')[-1].split(".")[:-1])
    if not (os.path.isdir('preprocessing/media/' + dir_for_save)):
        os.mkdir('preprocessing/media/' + dir_for_save)

    faces_paths = []
    for i, frame in enumerate(face_frames):
        face = Image.fromarray(frame)
        face.save('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
        faces_paths.append('preprocessing/media/' + dir_for_save + f"/{i}.jpg")

    return faces_paths


async def analyse_photo(filename):
    faces_paths = cut_faces(filename)
    # TODO add a response in case we haven't found faces
    data = {"data": faces_paths}
    results = await call_photo_inference(data)

    results["dir_path"] = 'preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1])
    # with zipfile.ZipFile('preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1]) + '.zip', 'w',
    #                      zipfile.ZIP_DEFLATED) as zipf:
    #     for root, dirs, files in os.walk('preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1])):
    #         for file in files:
    #             zipf.write(os.path.join(root, file))

    return results


async def call_photo_inference(data):
    url = "http://localhost:5050/api/v1/inference/photo"
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return {"message": "File analyzed successfully", "response": response.json()}
    else:
        return {"message": "Error sending request", "response": {}}


def analyse_audio(filename: str):
    return {"message": "Sorry, at this moment audio analysis is not available :(", "response": {}}


def calculate_features(image_path, model_session):
    # Загрузка изображения
    img = Image.open(image_path)
    img = img.resize((112, 112))
    # Преобразование изображения в массив данных
    arr_data = img2arr(img)
    # Загрузка модели
    ort_sess = model_session
    # Получение информации о входах модели
    input_info = ort_sess.get_inputs()
    # Подготовка данных для входа модели
    inputs = {}
    for inp in input_info:
        inputs[inp.name] = arr_data  # Преобразование массива данных в формат, понятный модели
    # Вычисление вектора признаков
    output = ort_sess.run(None, inputs)[0][0]
    return output


def compare_faces(existing_vec: list, face_path_to_compare: str, model_session):
    output = calculate_features(face_path_to_compare, model_session)

    best_id = -1
    best_vec = []
    best_prob = 0
    for i in range(len(existing_vec)):
        probability = cos_vec(output, existing_vec[i])

        if probability > 0.7 and probability > best_prob:
            best_id = i
            best_vec = existing_vec[i]
            best_prob = probability

    return best_id, best_vec


async def analyse_video(filename: str):
    # video = VideoFileClip(filename)
    # video.audio.write_audiofile(''.join(filename.split(".")[:-1]) + ".mp3")

    # result_audio = analyse_audio(''.join(filename.split(".")[:-1]) + ".mp3")  # for this moment it doesn't work

    ort_sess = ort.InferenceSession('preprocessing/photo_video/face_vec/vec_model.onnx')

    vidcap = cv2.VideoCapture(filename)
    result = {}
    vecs = []
    frame_num = 0
    while True:
        ret, frame = vidcap.read()  # what type is frame np.array
        if not ret:
            break
        faces_paths = cut_faces(filename, frame)  # DONE add an opportunity to upload np.array

        if not faces_paths:
            for face_id in list(result.keys()):
                result[str(face_id)].append(0)
            continue

        data = {"data": faces_paths}

        face_results = await call_photo_inference(data)

        if not result.keys():
            for i, face_path in enumerate(faces_paths):
                vec = calculate_features(face_path, ort_sess)
                vecs.append(vec)
                result[str(i)] = []
                result[str(i)].append(face_results["response"][str(i)])
            continue

        for i, face_path in enumerate(faces_paths):  # write properly
            face_id, face_vec = compare_faces(vecs, face_path, ort_sess)
            if face_id == -1:
                vecs.append(face_vec)
                face_id = len(vecs)
                result[str(face_id)] = []
                for _ in range(frame_num):
                    result[str(face_id)].append(0)
            result[str(face_id)].append(face_results["response"][str(i)])

        for face_path in faces_paths:
            os.remove(face_path)
        frame_num += 1

    return result  # {"message": "Sorry, at this moment video analysis is not available :(", "response": {}}
