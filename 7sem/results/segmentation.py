import os

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageOps import invert
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    return letters, letter_borders


def calc_profiles(input_image):
    profile_x = np.sum(input_image, 1)
    profile_y = np.sum(input_image, 0)

    return (profile_x, profile_y)
