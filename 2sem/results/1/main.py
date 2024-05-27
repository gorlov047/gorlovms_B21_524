import os

import numpy as np
from PIL import Image


working_dir = os.path.join(os.getcwd(), '2sem/results/1')


def fullcolor_to_halftone(image):
    height, width, channels = image.shape

    grayscale_image = np.mean(image, axis=2)  # averaging the channels

    halftone_image = np.zeros_like(image)

    for c in range(channels):
        halftone_image[:, :, c] = grayscale_image

    return halftone_image


def handle_img(img_name):
    fullcolor_image_path = working_dir + '/input/' + img_name
    fullcolor_image = np.array(Image.open(fullcolor_image_path))

    halftone_image = fullcolor_to_halftone(fullcolor_image)

    halftone_image_path = working_dir + '/output/' + img_name
    Image.fromarray(halftone_image.astype(np.uint8)).save(halftone_image_path)


def main():
    images = ('page1.png', 'page2.png', 'page3.png', 'page4.png')

    for img in images[1:]:
        handle_img(img)   


if __name__ == "__main__":
    main()
