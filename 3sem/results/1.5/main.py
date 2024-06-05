import numpy as np
from PIL import Image

BLACK, WHITE = 0, 255
OMEGA_OF_WHITE = 8 * WHITE


def logical_filtering(image_arr):
    res = np.zeros_like(image_arr, dtype=np.uint8)

    rows, columns = image_arr.shape
    
    res[0] = image_arr[0]
    res[~0] = image_arr[~0]
    for row in range(rows):
        res[row, 0] = image_arr[row, 0]
        res[row, -1] = image_arr[row, -1]

    for row in range(1, rows - 1):
        for col in range(1, columns - 1):
            big_omega = sum([image_arr[row + i, col + j] for i in range(-1, 2, 2) for j in range(-1, 2, 2)])
            if big_omega == BLACK and image_arr[row, col] == WHITE:
                res[row, col] = BLACK
            elif big_omega == OMEGA_OF_WHITE and image_arr[row, col] == BLACK:
                res[row, col] = WHITE
            else:
                res[row, col] = image_arr[row, col]
    return res


def main():
    images = ["img1.bmp", "img2.bmp"]

    for image in images:
        img_src = Image.open(f"3sem/results/1.5/input/{image}")
        image_arr = np.array(img_src, np.uint8)
        res_arr = logical_filtering(image_arr)

        Image.fromarray(res_arr).save(f"3sem/results/1.5/output/{image}")

        diff_arr = image_arr ^ res_arr
        Image.fromarray(diff_arr, 'L').save(f"3sem/results/1.5/output/xor_{image}")



if __name__ == "__main__":
    main()