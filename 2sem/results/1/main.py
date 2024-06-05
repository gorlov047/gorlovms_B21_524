from PIL import Image
import numpy as np


def semitone(old_img_arr):
    new_img_arr = 0.3 * old_img_arr[:,:,0] + 0.59 * old_img_arr[:,:,1] + 0.11 * old_img_arr[:,:,2]
    return new_img_arr.astype(np.uint8)


def main():
    images = ["198_115", "img1", "img2", "img3"]

    for image in images:
        img_src = Image.open(f"2sem/results/1/input/{image}.png").convert('RGB')
        src_image = semitone(np.array(img_src))
        img = Image.fromarray(src_image, 'L').convert('RGB')
        img.save(f"2sem/results/1/output/{image}.bmp")

if __name__ == "__main__":
    main()