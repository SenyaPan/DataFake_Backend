import onnxruntime as ort
import numpy as np
import torch
from PIL import Image


# Вычисление модуля вектора, не сильно нужно, так как все вектора длины 1
def len_vec(vec):
    return np.sqrt(np.sum(np.square(vec)))


# Нахождение косинуса угла между векторами и приведение данного значения в промежуток [0, 1]
def cos_vec(vec1, vec2):
    return (1 + (np.dot(vec1, vec2)) / (len_vec(vec1) * len_vec(vec2))) / 2


# Преобразование флатен списка байта в вид BGR и размерность [1, 3, 112, 112]
def convert_arr(arr, mode):
    conv_flt = []

    for chan in range(3):
        channel_data = arr[112 * 112 * chan:112 * 112 * (chan + 1)]
        channel = []
        for i in range(112):
            temp = []
            for j in range(112):
                temp.append(channel_data[112 * i + j])
            channel.append(temp)
        conv_flt.append(channel)
    if mode == 0:
        return torch.from_numpy(np.array([conv_flt]).astype(np.float32))
    elif mode == 1:
        return np.array([conv_flt]).astype(np.float32)


# Преобразование картинки в np.array размерности [1, 3, 112, 112]
def img2arr(img):
    x, y = img.size
    data = img.load()
    res = []
    for chan in range(2, -1, -1):
        channel = []
        for i in range(y):
            row = []
            for j in range(x):
                row.append(data[j, i][chan])
            channel.append(row)
        res.append(channel)
    res = (np.array([res]).astype(np.float32) - 128) / 128

    return res


# Преобразование np.array размерности [1, 3, 112, 112] в объект Image 
def arr2img(arr):
    w = 112
    h = 112
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    test_data = arr.flatten() * 128 + 128
    for y in range(h):
        for x in range(w):
            b = int(test_data[y * w + x])
            g = int(test_data[h * w + y * w + x])
            r = int(test_data[2 * h * w + y * w + x])
            pixels[x, y] = (r, g, b)

    return img


# Преобразуем bin файл в np.array размерности [1, 3, 112, 112]
def get_data_from_bin(bin_name):
    data = open(bin_name, "rb").read()
    flt = []
    w = 0x70
    h = 0x70
    for i in range(w * h * 3):
        d = data[4 * i:4 * (i + 1)]
        flt.append(unpack("<f", d)[0])

    return convert_arr(flt, 0)


# # Подгрузка модели
# ort_sess = ort.InferenceSession('./vec_model.onnx')
#
# img_data_name = ['3.png', '4.png']
# all_inputs = []
#
# outputs = []
#
# for el in img_data_name:
#     inputs = {}
#     for inp in ort_sess.get_inputs():
#         curshape = inp.shape
#         # print(f"{inp} - {curshape }")
#
#         curin = img2arr(Image.open(el))
#         inputs[inp.name] = curin
#     all_inputs.append(inputs)
#
# for el in all_inputs:
#     outputs.append(ort_sess.run(None, el)[0][0])
#
# depp = outputs[0]
# not_depp = outputs[1]
#
# # --------
#
# img_data_name = ['5.png']
#
# all_inputs = []
#
# outputs = [depp, not_depp]
#
# # Формируем выходы - для этой модели есть два входа, необходим словарь формата <название входа> : <значени>.
# # Делаем список из трех таких словарей в соответствии с загруженными изображениями
# for el in img_data_name:
#     inputs = {}
#     for inp in ort_sess.get_inputs():
#         curshape = inp.shape
#         # print(f"{inp} - {curshape }")
#
#         curin = img2arr(Image.open(el))
#         inputs[inp.name] = curin
#     all_inputs.append(inputs)
#
# # Формируем выходы
# for el in all_inputs:
#     outputs.append(ort_sess.run(None, el)[0][0])
#
# print("-----------")
# cos_vec_between_images = []
#
# # Сраниваем Депа с Джоли и с Депом соответственно
# for i in range(2, len(outputs)):
#     cos_vec_between_images.append(cos_vec(outputs[0], outputs[i]))
#
# # Выводим результат сравнения по углу между векторами
# print(cos_vec_between_images)
