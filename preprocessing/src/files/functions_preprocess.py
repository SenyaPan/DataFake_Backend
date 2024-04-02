from fastapi import APIRouter, Depends, UploadFile, Request
import hashlib
import magic


def check_extension(uploaded_file: UploadFile):
    return uploaded_file.filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"))


def check_file(file_path):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    if file_type.startswith('image/'):
        return 'image'
    elif file_type.startswith('audio/'):
        return 'audio'
    elif file_type.startswith('video/'):
        return 'video'
    else:
        return False


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
