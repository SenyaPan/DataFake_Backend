from fastapi import APIRouter, Depends, UploadFile, Request
import hashlib


def check_extension(uploaded_file: UploadFile):
    return uploaded_file.filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"))


def get_file_format(file_path: str):
    if file_path.split("/")[-1].split(".")[-1] in ["jpg", "jpeg", "png", "bmp"]:
        return "photo"
    elif file_path.split("/")[-1].split(".")[-1] in ["mp4", "avi", "mov"]:
        return "video"
    elif file_path.split("/")[-1].split(".")[-1] in ["mp3"]:  # add audio formats
        return "audio"


def get_hash_md5(filename: str):
    with open(filename, 'rb') as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
        return m.hexdigest()


def check_hash(file_hash):  # add type of params
    return False


def get_result(file_hash):
    return {}
