import os

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageOps import invert
import matplotlib.pyplot as plt
import matplotlib.patches as patches


working_dir = os.path.join(os.getcwd(), '6sem/results/')
input_path = f'{working_dir}/input'
output_path = f'{working_dir}/output'


def segment_letters(img, profile):
    letters = []
    letter_borders = []
    letter_start = 0
    is_empty = True

    for i in range(img.shape[1]):
        if profile[i] == 0:
            if not is_empty:
                is_empty = True
                letters.append(img[:, letter_start:i + 1])
                letter_borders.append(i+1)
        else:
            if is_empty:
                is_empty = False
                letter_start = i
                letter_borders.append(letter_start)

    letters.append(img[:, letter_start:img.shape[1] - 1])

    return letters, letter_borders


def bar(data, bins, axis):
    if axis == 1:
        plt.bar(x=bins, height=data)

    elif axis == 0:
        plt.barh(y=bins, width=data)


def calc_profiles_and_save(input_image):
    profile_x = np.sum(input_image, 1)
    profile_y = np.sum(input_image, 0)
    bins_x = np.arange(start=1, stop=input_image.shape[0] + 1).astype(int)
    bins_y = np.arange(start=1, stop=input_image.shape[1] + 1).astype(int)

    bar(profile_x / 255, bins_x, 0)
    plt.savefig(f'{output_path}/profile_x.png')
    plt.clf()

    bar(profile_y / 255, bins_y, 1)
    plt.savefig(f'{output_path}/profile_y.png')
    plt.clf()

    return (profile_x, profile_y)


def binarize_and_revert(img):
    img = np.where(img >= 32, 255, 0)  
    return 255 - img


def create_result_and_save_letters(input_image, img_letters, letter_borders):
    result_img = Image.fromarray(input_image.astype(np.uint8), 'L')
    rgb_img = Image.new("RGB", result_img.size)
    rgb_img.paste(result_img)
    draw = ImageDraw.Draw(rgb_img)
    
    for border in letter_borders:
        draw.line((border, 0, border, input_image.shape[1]), fill='green')

    rgb_img.save(f"{output_path}/result.png")

    for i, letter in enumerate(img_letters):
        letter_img = Image.fromarray(letter.astype(np.uint8), 'L').convert('1')
        letter_img.save(f"{output_path}/letter_{i}.png")


def handle_img(img_name):
    input_img_path = f'{input_path}/{img_name}'
    output_img_path = f'{output_path}/{img_name}'

    input_image = np.array(Image.open(input_img_path).convert('L'))
    input_image = binarize_and_revert(input_image)

    profile_x, profile_y = calc_profiles_and_save(input_image)
    
    img_letters, letter_borders = segment_letters(input_image, profile_y)

    create_result_and_save_letters(input_image, img_letters, letter_borders)


def main():
    handle_img('phrase_white.bmp')
        

if __name__ == "__main__":
    main()
