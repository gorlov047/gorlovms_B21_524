import os
import csv

import numpy as np
from numpy import mean
from math import pow, log, log2, floor
from PIL import Image
import matplotlib.pyplot as plt

from haralik import haralik
from features import AV, D


working_dir = os.path.join(os.getcwd(), '8sem/results/')
input_path = f'{working_dir}/input'
output_path = f'{working_dir}/output'


def image_to_np_array(image_name):
    img_src = Image.open(f'{input_path}/{image_name}').convert('RGB')
    return np.array(img_src)


def semitone(img):
    return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 *
            img[:, :, 2]).astype(np.uint8)


def to_semitone(img_name):
    img = image_to_np_array(img_name)
    return Image.fromarray(semitone(img), 'L')


def contrast(img):
    flat_img = img.flatten()
    mn = round(mean(flat_img))

    positiveRange = max(2, max(flat_img) - mn)
    negativeRange = max(2, mn - min(flat_img))
    
    positiveAlpha = 2 ** 7/ log(positiveRange)
    negativeAlpha = 2 ** 7/ log(negativeRange)

    res_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            f = img[i, j] - mn
            if f >= 1:
                res_img[i, j] = mn + positiveAlpha * log(f)
            elif f <= -1:
                res_img[i, j] = mn - negativeAlpha * log(abs(f))
            else:
                res_img[i, j] = mn

    return res_img


def main():
    img_names = ('kirp.png', 'oboi.png', 'sun.png')

    for img_name in img_names:
        semitone_img = to_semitone(img_name)
        semitone_img.save(f'{output_path}/semitone/{img_name}')
        semi = np.array(Image.open(f'{output_path}/semitone/{img_name}').convert('L'))

        transformed = contrast(semi)
        transformed_img = Image.fromarray(transformed.astype(np.uint8), "L")
        transformed_img.save(f'{output_path}/contrasted/{img_name}')
        
        figure, axis = plt.subplots(2, 1)
        axis[0].hist(x=semi.flatten(), bins=np.arange(0, 255))
        axis[0].title.set_text('Исходное изображение')

        axis[1].hist(x=transformed.flatten(), bins=np.arange(0, 255))
        axis[1].title.set_text('Преобразованное изображение')
        plt.tight_layout()
        plt.savefig(f'{output_path}/histograms/{img_name}')

        matrix = haralik(semi.astype(np.uint8))
        result = Image.fromarray(matrix.astype(np.uint8), "L")
        result.save(f'{output_path}/haralik/{img_name}')

        t_matrix = haralik(transformed.astype(np.uint8))
        t_result = Image.fromarray(t_matrix.astype(np.uint8), "L")
        t_result.save(f'{output_path}/haralik_contrasted/{img_name}')

        print('img_name:', img_name)

        print(f"AV: {AV(matrix)}")
        print(f"AV (contrasted): {AV(t_matrix)}")

        print(f"D: {D(matrix)}")
        print(f"D (contrasted): {D(t_matrix)}")

        print('###############\n')
    

if __name__ == "__main__":
    main()
