import time
from io import BytesIO

import cv2
import os
import requests
import shutil
import onnxruntime as ort

from tqdm import tqdm
from typing import Union

from PIL import Image

from preprocessing.photo_video.face_vec.feature_vec import img2arr, cos_vec

from preprocessing.src.config import pre_logger, face_extractor, ort_sess
from preprocessing.src.files.functions_audio import devide, create_spectrogram


# def cut_faces(filename, frame=None):
#     net = cv2.dnn.readNetFromCaffe('preprocessing/models/func_img_proc/deploy.prototxt',
#                                    'preprocessing/models/func_img_proc/weights.caffemodel')
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

async def cut_faces(filename, frame=None):
    dir_for_save = ''.join(filename.split('/')[-1].split(".")[:-1])

    if not (os.path.isdir('preprocessing/media/' + dir_for_save)):
        os.mkdir('preprocessing/media/' + dir_for_save)

    if frame is not None:
        try:
            boxes = face_extractor.process_file(frame)
            pre_logger.info(f'Frame from video {filename} has {len(boxes)} faces.')
        except:
            pre_logger.exception(f'During the face detecting in the video {filename} occurred an error.')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        dir_for_save += "/temp"
        if not os.path.isdir('preprocessing/media/' + dir_for_save):
            os.mkdir('preprocessing/media/' + dir_for_save)
    else:
        try:
            boxes = face_extractor.process_file(filename)
            pre_logger.info(f'The image {filename} has {len(boxes)} faces.')
        except:
            pre_logger.exception(f'During the face detecting in the image {filename} occurred an error.')
        image = Image.open(filename)

    faces_paths = []
    for i in range(len(boxes)):
        im = image.crop((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))
        try:
            im.save('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
            pre_logger.info(f'Face {i}.jpg was successfully saved in {dir_for_save} folder.')
        except:
            pre_logger.exception(f'During the save process of face {i}.jpg in {dir_for_save} folder occurred an error.')
        faces_paths.append('preprocessing/media/' + dir_for_save + f"/{i}.jpg")
    return faces_paths


async def analyse_photo(filename, model_num: Union[int, None]):
    faces_paths = await cut_faces(filename)

    results = {}
    for i, face_path in enumerate(faces_paths):
        with open(face_path, "rb") as f:
            results[i] = await call_photo_inference(f, model_num)

    # results["dir_path"] = 'preprocessing/media/' + ''.join(filename.split('/')[-1].split(".")[:-1])
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
    response = requests.post(url, files={'uploaded_file': data})

    if response.status_code == 200:
        return response.json()
    else:
        return {"message": "Error sending request", "response": {}}


async def analyse_audio(filename: str, sec: int = 1):
    audio_arr = devide(filename)
    result = []
    for sample in audio_arr:
        image = create_spectrogram(sample)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        response = await call_audio_inference(img_byte_arr)
        result.append(response)
    return result


async def calculate_features(image_path, model_session):
    # Загрузка изображения
    img = Image.open(image_path)
    img = img.resize((112, 112))
    # Преобразование изображения в массив данных
    arr_data = img2arr(img)
    # Получение информации о входах модели
    input_info = model_session.get_inputs()
    # Подготовка данных для входа модели
    inputs = {}
    for inp in input_info:
        inputs[inp.name] = arr_data  # Преобразование массива данных в формат, понятный модели
    # Вычисление вектора признаков
    output = model_session.run(None, inputs)[0][0]  # gets 2.8 seconds to be done
    return output


async def compare_faces(existing_vec: list, face_path_to_compare: str, model_session):
    output = await calculate_features(face_path_to_compare, model_session)

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

    vidcap = cv2.VideoCapture(filename)
    result = {}
    vecs = []
    frame_num = 0
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # with tqdm(total=total_frames) as pbar:
    while True:
        ret, frame = vidcap.read()

        if not ret:
            break

        if frame_num % 4 != 0:
            frame_num += 1
            # pbar.update(1)
            continue
        pre_logger.info(f"#{frame_num} frame from video {filename} is analyzed.")
        # start_time = time.time()
        faces_paths = await cut_faces(filename, frame)
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
                vec = await calculate_features(face_path, ort_sess)
                # end_time = time.time()
                # print("Count faces vecs time is ", end_time - start_time)
                vecs.append(vec)
                result[str(i)] = []
                result[str(i)].append(model_results[i])
                move_to = ''.join(filename.split(".")[:-1])
                try:
                    shutil.move(face_path, move_to)
                    pre_logger.info(f'The face {i} moved successfully from {face_path} to {move_to}')
                except:
                    pre_logger.exception(f'During moving {face_path} to {move_to} an error occurred.')
            continue

        found_face_id = []
        for i, face_path in enumerate(faces_paths):  # write properly
            # start_time = time.time()
            face_id, face_vec = await compare_faces(vecs, face_path, ort_sess)
            # end_time = time.time()
            # print("Compare faces time is ", end_time - start_time)
            if face_id == -1:
                face_id = len(vecs)
                vecs.append(face_vec)
                result[str(face_id)] = []
                move_to = ''.join(filename.split(".")[:-1]) + f'/{face_id}.jpg'
                try:
                    shutil.move(face_path, move_to)
                    pre_logger.info(f'The face {i} moved successfully from {face_path} to {move_to}')
                except:
                    pre_logger.exception(f'During moving {face_path} to {move_to} an error occurred.')
                for _ in range(frame_num):
                    result[str(face_id)].append(None)
            found_face_id.append(str(face_id))
            result[str(face_id)].append(model_results[i])
        if set(found_face_id) != set(result.keys()):
            for face_id in list(result.keys()):
                if face_id not in set(found_face_id):
                    result[face_id].append(None)
        frame_num += 1
        # pbar.update(1)
    rm_tree = ''.join(filename.split('.')[:-1]) + '/temp'
    try:
        shutil.rmtree(rm_tree)
        pre_logger.info(f'The folder {rm_tree} removed successfully.')
    except:
        pre_logger.info(f'During removing the folder {rm_tree} an error occurred.')
    right_result = {'message': "File analyzed successfully", 'response': result}
    # add to the result audio_result
    return right_result  # {"message": "Sorry, at this moment video analysis is not available :(", "response": {}}
