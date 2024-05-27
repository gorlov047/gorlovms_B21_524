import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


working_dir = os.path.join(os.getcwd(), '1sem/results/1.2')


def decimation(image, scale_factor):
    downsampled_image = image[::scale_factor, ::scale_factor]  # takes nth row and column from original img

    return downsampled_image


def handle_img(img_name):
    source_image_path = working_dir + '/input/' + img_name
    source_image = np.array(Image.open(source_image_path))

    downsampled_image = decimation(source_image, scale_factor=2)

    downsampled_image_path = working_dir + "/output/" + img_name
    Image.fromarray(downsampled_image).save(downsampled_image_path)


def main():
    images = ('kotik.png', 'muar.png', 'scls.png')

    for img in images:
        handle_img(img)   


if __name__ == "__main__":
    main()
