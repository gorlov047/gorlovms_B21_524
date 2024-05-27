from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy.signal import convolve2d

original_image = Image.open("original_image.png")

grayscale_image = ImageOps.grayscale(original_image)
grayscale_image.save("grayscale_image.png")

grayscale_array = np.array(grayscale_image)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gradient_x = convolve2d(grayscale_array, sobel_x, mode='same')
gradient_y = convolve2d(grayscale_array, sobel_y, mode='same')

gradient = np.sqrt(gradient_x**2 + gradient_y**2)
gradient = gradient.clip(0, 255).astype(np.uint8) 

threshold = 100 
binary_gradient = gradient > threshold

Image.fromarray(gradient_x.astype(np.uint8)).save("gradient_x.png")
Image.fromarray(gradient_y.astype(np.uint8)).save("gradient_y.png")
Image.fromarray(gradient).save("gradient.png")
Image.fromarray(binary_gradient.astype(np.uint8) * 255).save("binary_gradient.png")

Image.fromarray(gradient_x.astype(np.uint8)).show()
Image.fromarray(gradient_y.astype(np.uint8)).show()
Image.fromarray(gradient).show()
Image.fromarray(binary_gradient.astype(np.uint8) * 255).show()