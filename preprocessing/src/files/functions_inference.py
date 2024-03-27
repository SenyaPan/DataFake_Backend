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


def compare_faces(existing_vec: list, face_path_to_compare: str):
    ort_sess = ort.InferenceSession('preprocessing/photo_video/face_vec/vec_model.onnx')

    input_face = {}
    for inp in ort_sess.get_inputs():
        curin = img2arr(Image.open(face_path_to_compare))
        input_face[inp.name] = curin

    output = ort_sess.run(None, input_face)[0][0]

    best_id = -1
    best_prob = 0
    for i in range(len(existing_vec)):
        probability = cos_vec(output, existing_vec[i])
        if probability > 0.7 and probability > best_prob:
            best_id = i
            best_prob = probability

    return best_id


async def analyse_video(filename: str):
    # video = VideoFileClip(filename)
    # video.audio.write_audiofile(''.join(filename.split(".")[:-1]) + ".mp3")

    # result_audio = analyse_audio(''.join(filename.split(".")[:-1]) + ".mp3")  # for this moment it doesn't work

    vidcap = cv2.VideoCapture(filename)
    result = {}
    frame_num = 0
    while True:
        ret, frame = vidcap.read()  # what type is frame np.array
        if not ret:
            break
        faces_paths = cut_faces(filename, frame)  # DONE add an opportunity to upload np.array

        if not faces_paths:
            for i in range(len(result.keys())):
                result[i].append(0)
            continue

        data = {"data": faces_paths}

        face_results = await call_photo_inference(data)

        if not result.keys():
            for i in range(len(faces_paths)):
                result[i] = [face_results["response"][str(i)]]
            continue
        for i, face_path in enumerate(faces_paths):  # write properly
            face_id = compare_faces(result.keys(), face_path)
            if face_id == -1:
                face_id = len(result.keys())
                result[face_id] = []
                for _ in range(frame_num):
                    result[face_id].append(0)
            result[face_id].append(face_results["response"][str(i)])

        for face_path in faces_paths:
            os.remove(face_path)
        frame_num += 1

    return result  # {"message": "Sorry, at this moment video analysis is not available :(", "response": {}}
