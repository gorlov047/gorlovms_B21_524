import numpy as np
from numpy import log
from scipy.ndimage import convolve


# def haralik(image):
#     filters = [np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
#                np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
#                np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]]),
#                np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])]

#     haralick_matrix = np.zeros((256, 256))

#     for filter in filters:
#         filtered_image = convolve(image, filter)
#         hist, _ = np.histogram(filtered_image, bins=range(257))
#         haralick_matrix += np.outer(hist, hist)

#     return haralick_matrix


def haralik(img_arr, d = 2):
    matrix = np.zeros(shape=(256, 256))

    for x in range(d, img_arr.shape[0] - d):
        for y in range(d, img_arr.shape[1] - d):
            matrix[img_arr[x - d, y], img_arr[x, y]] += 1
            matrix[img_arr[x + d, y], img_arr[x, y]] += 1
            matrix[img_arr[x, y - d], img_arr[x, y]] += 1
            matrix[img_arr[x, y + d], img_arr[x, y]] += 1

    for x in range(256):
        m = np.array(matrix[x])
        m[np.where(m == 0)] = 1
        matrix[x] = log(m)
        
    matrix = matrix * 256 / np.max(matrix)
    return matrix
