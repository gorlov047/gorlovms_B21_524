from PIL import Image
import numpy as np


def semitone(old_img_arr):
    new_img_arr = 0.3 * old_img_arr[:,:,0] + 0.59 * old_img_arr[:,:,1] + 0.11 * old_img_arr[:,:,2]
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

1) Для каждого пикселя изображения вычисляется среднее значение интенсивности в его окрестности,
она задана квадратным или круглым окном какого-то размера

2) Вычисляется весовой коэффициент для каждого пикселя в окрестности, 
зависит от  расстояния до центрального пикселя или разности интенсивностей

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


def main():
    images = ["img1.bmp", "img2.bmp", "img3.bmp"]

    for image in images:
        img_src = Image.open(f"2sem/results/2.8/input/{image}").convert('RGB')
        gray_img_arr = semitone(np.array(img_src))
        binarized_image = wan_binarization(gray_img_arr)
        img = Image.fromarray(binarized_image)
        img.save(f"2sem/results/2.8/output/{image}")


if __name__ == "__main__":
    main()