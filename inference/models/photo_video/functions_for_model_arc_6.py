import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from scipy.fft import dctn


def noise_proc(image):
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image).unsqueeze(0)  # Добавление batch dimension

    # Определение ядра фильтра
    filter_kernel = [[-1, 2, -2, 2, -1],
                     [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2],
                     [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]]
    kernel = np.array(filter_kernel, dtype=np.float32) / 0.256
    kernel_tensor = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)  # Добавление channel и batch dimensions

    # Применение фильтра ко всем каналам изображения
    kernel_tensor = kernel_tensor.repeat(image_tensor.shape[1], 1, 1, 1)
    filtered_image_tensor = F.conv2d(image_tensor, kernel_tensor, padding=2, groups=image_tensor.shape[1])

    # Удаление batch dimension
    filtered_image_tensor = filtered_image_tensor.squeeze(0)

    # Изменение размера отфильтрованного тензора до 224x224
    transform_resize = transforms.Resize((224, 224))
    resized_filtered_image_tensor = transform_resize(filtered_image_tensor)

    return resized_filtered_image_tensor


def fft_proc(image):
    ycbcr_array = np.array(image.convert('YCbCr'))
    y_component = ycbcr_array[:,:,0]

    dct_coefficients = dctn(y_component.astype(float), norm='ortho')

    magnitude_spectrum = np.log(np.abs(dct_coefficients) + 1)  # Добавляем 1, чтобы избежать логарифма от нуля

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Преобразование в PIL изображение для использования Resize
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # Преобразование в тензор
    ])
    resized_magnitude_tensor = transform(magnitude_spectrum)

    result_tensor = resized_magnitude_tensor

    return result_tensor