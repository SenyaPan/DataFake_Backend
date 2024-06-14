import numpy as np
from tqdm import tqdm
from PIL import Image
import librosa
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa.display as display

import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


def devide(path_to_audio, win=16000, hop=8000):
    y, sr = librosa.load(path_to_audio, sr=16000)
    pos = 0
    arr = []
    np_arr = np.ndarray.tolist(y)
    while pos <= len(np_arr) - win:
        arr.append(np.array(np_arr[pos:pos + win]))
        pos += hop
        if (pos > len(np_arr) - win) and (len(np_arr) - pos - hop > hop):
            arr.append(np.array(np_arr[pos:len(np_arr)], dtype='float32'))
    return arr


def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    return image


def create_spectrogram(audio):
    plt.interactive(False)
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    ax.axis('off')
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, hop_length=256, win_length=512)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max))
    plt.axis("off")
    img = fig2data(fig)
    plt.close()
    # fig.clf()
    plt.close(fig)
    plt.close('all')
    return img



