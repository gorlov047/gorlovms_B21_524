import os
import csv

import numpy as np
from PIL import Image

from segmentation import calc_profiles, segment_letters
from features_computer import FeaturesComputer


working_dir = os.path.join(os.getcwd(), '7sem/results/')
input_path = f'{working_dir}/input'
output_path = f'{working_dir}/output'


def open_img_and_prepare(input_img_path):
    img = np.array(Image.open(input_img_path).convert('L'))
    img = np.where(img >= 54, 255, 0)  

    # threshold=54 for phrase 10
    # threshold=128 for phrase 14 

    return 255 - img


def calc_distance_between_letter_and_alphabet(img_letter, alphabet, letter_in_sentence):
    def get_letter_features(letter):
        return [
            FeaturesComputer.get_normalized_quarter_masses(letter),            # масса 
            FeaturesComputer.get_normalized_center_of_mass(letter),            # координаты центра тяжести
            FeaturesComputer.get_normalized_axial_moments_of_inertia(letter),  # осевые моменты инерции
        ]
    
    def calc_similarity(img_letter_features, alphabet_letter_features):
        return round(
            sum(
                img_letter_features[i].calc_distance(alphabet_letter_features[i])
                for i in range(3)
            ) ** .5, 
            3
        )
    
    def calc_normalized_value(standardized_value, x):
        if x > standardized_value:
            return round(standardized_value / x, 3)
        return round(x / standardized_value, 3)

    try:
        index_of_letter_alphabet = alphabet.index(letter_in_sentence) + 1
    except ValueError as e: # letter doesn't exist in the sentence
        return None

    standardized_letter_img = open_img_and_prepare(f'{input_path}/letters/{index_of_letter_alphabet}.png')

    standardized_features = get_letter_features(standardized_letter_img)

    img_letter_features = get_letter_features(img_letter)
    
    standardized_value = calc_similarity(standardized_features, img_letter_features)

    results = []

    for i, alphabet_letter in enumerate(alphabet):
        alphabet_letter_image = open_img_and_prepare(f'{input_path}/letters/{i+1}.png')
        alphabet_letter_features = get_letter_features(alphabet_letter_image)
        
        similarity = calc_similarity(img_letter_features, alphabet_letter_features)

        normalized_similarity = calc_normalized_value(standardized_value, similarity)

        results.append((alphabet_letter, normalized_similarity))

    results.sort(key=lambda tpl: tpl[-1], reverse=True)

    return results


def save_results_to_csv(results_by_letter):
    with open(f'{output_path}/result.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in results_by_letter:
            writer.writerow(row)


def main():
    img_name = 'phrase.bmp'
    input_img_path = f'{input_path}/{img_name}'
    output_img_path = f'{output_path}/{img_name}'
    alphabet = 'աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքևօֆ'
    sentence = 'սերը բոլոր տարիքի հնազանդ է'.replace(' ', '')
    
    input_image = open_img_and_prepare(input_img_path)

    profile_x, profile_y = calc_profiles(input_image)
    
    letters, letter_borders = segment_letters(input_image, profile_y)

    results_by_letter = []

    # print(f'{len(letters)=}', f'{len(sentence)=}')

    for i, letter in enumerate(letters):
        results_by_letter.append(
            calc_distance_between_letter_and_alphabet(letter, alphabet, sentence[i])
        )

    save_results_to_csv(results_by_letter)


if __name__ == "__main__":
    main()