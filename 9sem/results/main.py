import os
import csv

from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

from utils import spectrogram_plot, denoise, to_pcm


working_dir = os.path.join(os.getcwd(), '9sem/results/')
input_path = f'{working_dir}/input'
output_path = f'{working_dir}/output'


def main():
    file_name = 'audio_test.wav'

    dpi = 500
    sample_rate, samples = wavfile.read(f'{input_path}/{file_name}')
    plt.figure(dpi=dpi)
  
    spectrogram_plot(samples, sample_rate, 20000)
    plt.savefig(f'{output_path}/spectrogram.png', dpi = dpi)
    plt.clf()

    # фильтр Савицкого-Голея
    denoised_0 = denoise(samples, sample_rate, cutoff_freuency = 3000, passes = 0)
    spectrogram_plot(denoised_0, sample_rate, 20000)
    plt.savefig(f'{output_path}/denoised_spectrogram_savgol.png', dpi = dpi) 
    plt.clf()

    # фильтр нижних частот
    denoised = denoise(samples, sample_rate, cutoff_freuency = 3000)
    spectrogram_plot(denoised, sample_rate)
    plt.savefig(f'{output_path}/denoised_spectrogram_once.png', dpi = dpi)  
    plt.clf()

    wavfile.write(f'{output_path}/denoised_once.wav', sample_rate, to_pcm(denoised))

    denoised_2 = denoise(samples, sample_rate, cutoff_freuency = 3000, passes = 2)
    spectrogram_plot(denoised_2, sample_rate)
    plt.savefig(f'{output_path}/denoised_spectrogram_twice.png', dpi = dpi)
    plt.clf()

    wavfile.write(f'{output_path}/denoised_twice.wav', sample_rate, to_pcm(denoised_2))
    

if __name__ == "__main__":
    main()
