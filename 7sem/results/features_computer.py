import numpy as np

from utils import NPoint


class FeaturesComputer:
    @classmethod
    def _get_quarter_masses(cls, image):
        height, width = image.shape

        half_height = height // 2
        half_width = width // 2

        top_left = round(np.sum(image[:half_height, :half_width]), 3)
        top_right = round(np.sum(image[:half_height, half_width:]), 3)
        bottom_left = round(np.sum(image[half_height:, :half_width]), 3)
        bottom_right = round(np.sum(image[half_height:, half_width:]), 3)

        return (top_left, top_right, bottom_left, bottom_right)

    @classmethod
    def get_normalized_quarter_masses(cls, image):
        quarter_masses = cls._get_quarter_masses(image)
        height, width = image.shape
        quarter_area = (height // 2) * (width // 2)

        normalized_tpl = tuple(round(k / quarter_area, 3) for k in quarter_masses)

        return NPoint(*normalized_tpl)

    @classmethod
    def _get_center_of_mass(cls, image):
        black_pixel_indices = np.argwhere(image == 0)
        
        center_y = round(np.mean(black_pixel_indices[:, 0]), 3)
        center_x = round(np.mean(black_pixel_indices[:, 1]), 3)
        
        return center_x, center_y

    @classmethod
    def get_normalized_center_of_mass(cls, image):
        center_x, center_y = cls._get_center_of_mass(image)
        height, width = image.shape
        
        normalized_center_x = round(center_x / width, 3)
        normalized_center_y = round(center_y / height, 3)
        
        return NPoint(normalized_center_x, normalized_center_y)

    @classmethod
    def _get_axial_moments_of_inertia(cls, image):
        center_x, center_y = cls._get_center_of_mass(image)

        black_pixel_indices = np.argwhere(image == 0)

        horizontal_distances = black_pixel_indices[:, 1] - center_x
        vertical_distances = black_pixel_indices[:, 0] - center_y

        moment_x = round(np.sum(horizontal_distances ** 2), 3)
        moment_y = round(np.sum(vertical_distances ** 2), 3)

        return (moment_x, moment_y)

    @classmethod
    def get_normalized_axial_moments_of_inertia(cls, image):
        height, width = image.shape
        moment_x, moment_y = cls._get_axial_moments_of_inertia(image)

        normalized_moment_x = round(moment_x / (width ** 2 * height ** 2), 3)
        normalized_moment_y = round(moment_y / (height ** 2 * width ** 2), 3)

        return NPoint(normalized_moment_x, normalized_moment_y)
