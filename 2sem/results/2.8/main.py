import os

from PIL import Image
import numpy as np


working_dir = os.path.join(os.getcwd(), '2sem/results/2.8')

def semitone(old_img_arr):
    height = old_img_arr.shape[0]
    width = old_img_arr.shape[1]

    new_img_arr = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = old_img_arr[y, x]
            new_img_arr[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img_arr.astype(np.uint8)


def compute_local_threshold(image, window_size=15, C=10):
    """
    Вычисление локального порога для каждого пикселя.
    """
    height, width = image.shape
    padded_image = np.pad(image, pad_width=window_size // 2, mode='edge')
    threshold_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            local_area = padded_image[i:i + window_size, j:j + window_size]
            local_mean = np.mean(local_area)
            threshold_image[i, j] = local_mean - C

    return threshold_image


"""
Алгоритм WAN (Weighted Adaptive Neighborhood) работает следующим образом:

1) Для каждого пикселя изображения вычисляется среднее значение интенсивности в его окрестности. 
Эта окрестность может быть задана, например, квадратным или круглым окном определенного размера.

2) Вычисляется весовой коэффициент для каждого пикселя в окрестности, 
который может зависеть от различных факторов, таких как расстояние до центрального пикселя или 
разность интенсивностей.

3) На основе среднего значения интенсивности и весовых коэффициентов вычисляется адаптивный порог для 
каждого пикселя. Пиксели, чья интенсивность выше порога, становятся белыми, а те, что ниже - черными.
"""
def wan_binarization(image, window_size=15, C=10):
    """
    Адаптивная бинаризация изображения методом WAN.

    :param image: Исходное изображение в градациях серого.
    :param window_size: Размер окна для вычисления адаптивного порога.
    :param C: Константа, вычитаемая из среднего значения для определения порога.
    :return: Бинаризованное изображение.
    """
    if image.ndim == 3:
        image = semitone(image).astype(np.uint8)

    threshold_image = compute_local_threshold(image, window_size, C)
    binarized_image = np.where(image > threshold_image, 255, 0).astype(np.uint8)

    return binarized_image

def handle_img(img_name):
    fullcolor_image_path = working_dir + '/input/' + img_name
    fullcolor_image = np.array(Image.open(fullcolor_image_path))

    halftone_image = wan_binarization(fullcolor_image)

    halftone_image_path = working_dir + '/output/' + img_name
    Image.fromarray(halftone_image.astype(np.uint8)).save(halftone_image_path)


def main():
    images = ("pag2.png")

    for img in images:
        handle_img(img)   

if __name__ == "__main__":
    main()