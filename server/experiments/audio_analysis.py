from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():
    sample_rate, data = wavfile.read('audio.wav')

    N = len(data)
    spectrum = fft(data[:N])
    frequencies = fftfreq(N, 1/sample_rate)[:N//2]

    plt.plot(frequencies, 2.0/N * np.abs(spectrum[0:N//2]))
    plt.xlim([18000, 20500])
    plt.ylim([0, 1])
    plt.savefig('audio_analysis_result_zoom.jpg')
    plt.clf()

    plt.plot(frequencies, 2.0/N * np.abs(spectrum[0:N//2]))
    plt.savefig('audio_analysis_result.jpg')


if __name__ == "__main__":
    main()
