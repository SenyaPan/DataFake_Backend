import cv2
import os
import librosa
import requests
import shutil
import time
import onnxruntime as ort
import numpy as np

from tqdm import tqdm
from typing import Union

from matplotlib import pyplot as plt

import torch

from ultralytics import YOLO
from PIL import Image

from PIL import Image
from fastapi.responses import JSONResponse, FileResponse

from preprocessing.photo_video.func_img_proc.face_crop_2 import FaceExtractor
from preprocessing.photo_video.face_vec.feature_vec import img2arr, cos_vec


# def cut_faces(filename, frame=None):
#     net = cv2.dnn.readNetFromCaffe('preprocessing/photo_video/func_img_proc/deploy.prototxt',
#                                    'preprocessing/photo_video/func_img_proc/weights.caffemodel')
#     face_extractor = FaceExtractor(net)
#
#     if frame is None:
#         face_frames = face_extractor.process_file(filename)
#     else:
#         face_frames = face_extractor.process_file(frame)
#
#     dir_for_save = ''.join(filename.split('/')[-1].split(".")[:-1])
#
#     if not (os.path.isdir('preprocessing/media/' + dir_for_save)):
#         os.mkdir('preprocessing/media/' + dir_for_save)
#
#     if filename.split("/")[-1].split(".")[-1].lower() in ["mp4", "avi", "mov"]:
#         dir_for_save += '/temp'
#
#     if not (os.path.isdir('preprocessing/media/' + dir_for_save)):
#         os.mkdir('preprocessing/media/' + dir_for_save)
#
#     faces_paths = []
#     for i, frame in enumerate(face_frames):
#         face = Image.fromarray(frame)
#         face.save('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
#         faces_paths.append('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
#
#     return faces_paths

def cut_faces(filename, frame=None):
    model_path = 'preprocessing/photo_video/func_img_proc/model.pt'
    model = YOLO(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    face_extractor = FaceExtractor(model)

    dir_for_save = ''.join(filename.split('/')[-1].split(".")[:-1])

    if not (os.path.isdir('preprocessing/media/' + dir_for_save)):
        os.mkdir('preprocessing/media/' + dir_for_save)

    if frame is not None:
        boxes = face_extractor.process_file(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        dir_for_save += "/temp"
        if not os.path.isdir('preprocessing/media/' + dir_for_save):
            os.mkdir('preprocessing/media/' + dir_for_save)
    else:
        boxes = face_extractor.process_file(filename)
        image = Image.open(filename)

    faces_paths = []
    for i in range(len(boxes)):
        im = image.crop((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))
        im.save('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
        faces_paths.append('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
    return faces_paths


async def analyse_photo(filename, model_num: Union[int, None]):
    faces_paths = cut_faces(filename)

    results = {}
    for i, face_path in enumerate(faces_paths):
        with open(face_path, "rb") as f:
            results[i] = await call_photo_inference(f, model_num)

    #results["dir_path"] = 'preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1])
    # with zipfile.ZipFile('preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1]) + '.zip', 'w',
    #                      zipfile.ZIP_DEFLATED) as zipf:
    #     for root, dirs, files in os.walk('preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1])):
    #         for file in files:
    #             zipf.write(os.path.join(root, file))
    response = {'message': 'File analyzed successfully', 'response': results}

    return response


async def call_photo_inference(file, model_num: Union[int, None]):
    if model_num is None:
        url = "http://localhost:5050/api/v1/inference/photo"
    else:
        url = f"http://localhost:5050/api/v1/inference/photo?model_num={model_num}"
    response = requests.post(url, files={'uploaded_file': file})

    if response.status_code == 200:
        return response.json()
    else:
        return {"message": "Error sending request"}


async def call_audio_inference(data):
    url = "http://localhost:5050/api/v1/inference/audio"
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return {"message": "File analyzed successfully", "response": response.json()}
    else:
        return {"message": "Error sending request", "response": {}}


def create_spectrogram(audio, sample_rate, name):
    plt.interactive(False)
    fig = plt.figure(figsize=[0.715, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = name + '.png'
    plt.savefig(filename, dpi=405, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del name, audio, sample_rate, fig, ax, S


def analyse_audio(filename: str, sec: int = 1):
    audio, sr = librosa.load(filename)
    buffer = sec * sr  # число-количесвто секунд в отрезке
    samples_total = len(audio)
    samples_wrote = 0
    counter = 1
    list_of_audio = []
    audio_path = ''.join(filename.split('.')[:-1])
    audio_result = []
    for i in range(0, samples_total, buffer):

        if samples_total - buffer < i:
            create_spectrogram(audio[i:samples_total], sr, audio_path + '0.png')
        else:
            create_spectrogram(audio[i:i + buffer], sr, audio_path + '0.png')

        data = {"data": audio_path + '0.png'}

        result = call_audio_inference(data)

        audio_result.append(result)

    right_result = {'message': "File analyzed successfully", 'response': audio_result, 'dir_path': ''}

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

    if best_id == -1:
        best_vec = output

    return best_id, best_vec


def get_ort_session(model_path):
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        # GPU is available, use it
        return ort.InferenceSession(model_path, providers=['CUDAExecutionProvider']), print('Using CUDA')
    else:
        # GPU is not available, use CPU
        return ort.InferenceSession(model_path)


async def analyse_video(filename: str, model_num: Union[int, None]):

    # video = VideoFileClip(filename)
    # video.audio.write_audiofile(''.join(filename.split(".")[:-1]) + ".mp3")

    # result_audio = analyse_audio(''.join(filename.split(".")[:-1]) + ".mp3")  # for this moment it doesn't work

    # video = VideoFileClip(filename)
    # audio = video.audio
    # video.close()
    #
    # audio_path = ''.join(filename.split('.')[-1]) + '0.mp3'
    # audio.write_audiofile(audio_path)
    # audio.close()
    #
    # result_audio = analyse_audio(audio_path)

    model_path = 'preprocessing/photo_video/face_vec/vec_model.onnx'

    ort_sess = get_ort_session(model_path)

    vidcap = cv2.VideoCapture(filename)
    result = {}
    vecs = []
    frame_num = 0
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = vidcap.read()

            if not ret:
                break

            if frame_num % 4 != 0:
                frame_num += 1
                pbar.update(1)
                continue
            # start_time = time.time()
            faces_paths = cut_faces(filename, frame)
            # end_time = time.time()
            # print("Cut faces time is ", end_time - start_time)

            if not faces_paths:
                for face_id in list(result.keys()):
                    result[str(face_id)].append(None)
                continue

            model_results = {}
            for i, face_path in enumerate(faces_paths):
                with open(face_path, "rb") as f:
                    model_results[i] = await call_photo_inference(f, model_num)

            if not result.keys():
                for i, face_path in enumerate(faces_paths):
                    # start_time = time.time()
                    vec = calculate_features(face_path, ort_sess)
                    # end_time = time.time()
                    # print("Count faces vecs time is ", end_time - start_time)
                    vecs.append(vec)
                    result[str(i)] = []
                    result[str(i)].append(model_results[i])
                    shutil.move(face_path, ''.join(filename.split(".")[:-1]))
                continue

            found_face_id = []
            for i, face_path in enumerate(faces_paths):  # write properly
                # start_time = time.time()
                face_id, face_vec = compare_faces(vecs, face_path, ort_sess)
                # end_time = time.time()
                # print("Compare faces time is ", end_time - start_time)
                if face_id == -1:
                    face_id = len(vecs)
                    vecs.append(face_vec)
                    result[str(face_id)] = []
                    shutil.move(face_path, ''.join(filename.split(".")[:-1]) + f'{face_id}.jpg')
                    for _ in range(frame_num):
                        result[str(face_id)].append(None)
                found_face_id.append(str(face_id))
                result[str(face_id)].append(model_results[i])
            if set(found_face_id) != set(result.keys()):
                for face_id in list(result.keys()):
                    if face_id not in set(found_face_id):
                        result[face_id].append(None)
            frame_num += 1
            pbar.update(1)
    shutil.rmtree(''.join(filename.split('.')[:-1]) + '/temp')
    right_result = {'message': "File analyzed successfully", 'response': result}  # ,
                    # 'dir_path': 'preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1])}
    # add to the result audio_result
    return right_result  # {"message": "Sorry, at this moment video analysis is not available :(", "response": {}}
