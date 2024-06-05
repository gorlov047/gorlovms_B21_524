from PIL import Image, ImageOps
import numpy as np
from scipy.signal import convolve2d

def main():
    img = Image.open("4sem/results/5/input/img.png")

    grayscale_image = ImageOps.grayscale(img)
    grayscale_image.save("4sem/results/5/output/grayscale_image.png")

    grayscale_array = np.array(grayscale_image)

    sharra_x = np.array([[3, 0, -3],
                        [10, 0, -10],
                        [3, 0, -3]])
    
    sharra_y = np.array([[3, 10, 3],
                        [0, 0, 0],
                        [-3, -10, -3]])

    gradient_x = convolve2d(grayscale_array, sharra_x, mode='same')
    gradient_y = convolve2d(grayscale_array, sharra_y, mode='same')

    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = gradient.clip(0, 255).astype(np.uint8) 

    threshold = 250
    binary_gradient = gradient > threshold

    Image.fromarray(gradient_x.astype(np.uint8)).save("4sem/results/5/output/gradient_x.png")
    Image.fromarray(gradient_y.astype(np.uint8)).save("4sem/results/5/output/gradient_y.png")
    Image.fromarray(gradient).save("4sem/results/5/output/gradient.png")
    Image.fromarray(binary_gradient.astype(np.uint8) * 255).save("4sem/results/5/output/binary_gradient.png")


if __name__ == "__main__":
    main()



