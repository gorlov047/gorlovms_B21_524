import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


working_dir = os.path.join(os.getcwd(), '1sem/results/1.1')


def bilinear_interpolation(image, scale_factor):
    height, width = image.shape[:2]

    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    interpolated_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    x_scale = float(width - 1) / (new_width - 1)
    y_scale = float(height - 1) / (new_height - 1)

    for y in range(new_height):
        for x in range(new_width):
            x_original = x * x_scale
            y_original = y * y_scale
            x_floor, y_floor = int(x_original), int(y_original)
            x_ceil, y_ceil = min(x_floor + 1, width - 1), min(y_floor + 1, height - 1)
            x_weight = x_original - x_floor
            y_weight = y_original - y_floor

            top_left = image[y_floor, x_floor]
            top_right = image[y_floor, x_ceil]
            bottom_left = image[y_ceil, x_floor]
            bottom_right = image[y_ceil, x_ceil]

            interpolated_pixel = (1 - x_weight) * (1 - y_weight) * top_left + \
                                 x_weight * (1 - y_weight) * top_right + \
                                 (1 - x_weight) * y_weight * bottom_left + \
                                 x_weight * y_weight * bottom_right

            interpolated_image[y, x] = interpolated_pixel.astype(np.uint8)

    return interpolated_image


def handle_img(img_name):
    source_image_path = working_dir + '/input/' + img_name
    source_image = np.array(Image.open(source_image_path))

    interpolated_image = bilinear_interpolation(source_image, scale_factor=2)

    interpolated_image_path = working_dir + "/output/" + img_name
    Image.fromarray(interpolated_image).save(interpolated_image_path)


def main():
    images = ('kotik.png', 'muar.png', 'scls.png')

    for img in images:
        handle_img(img)   


if __name__ == "__main__":
    main()
