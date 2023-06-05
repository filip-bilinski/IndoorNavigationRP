from scipy.signal import spectrogram
from scipy.signal.windows import hann
import numpy as np
import scipy.io.wavfile 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import time




class globals:
    def __init__(self):
        self.min_frequency = 19500
        self.max_frequency = 20500

        self.sample_rate = 44100
        self.frequency = 20000
        self.duration = 0.02 

        self.interval = 0.1
        self.chirp_duration = 0.0025
        self.interval_samples = self.sample_rate * self.interval
        self.chirp_amount = 507
        self.cutoff = 0.0138 * self.sample_rate

        self.chirp_radius_samples = int(self.sample_rate * self.chirp_duration/2)

params = globals()

def debug_spectrogram(array, filename):
    
    f, t, Sxx = spectrogram(array, params.sample_rate, window=hann(256, sym=False))

    
    indecies = np.where((f > params.min_frequency) & (f < params.max_frequency))

    
    Sxx = Sxx[indecies]
    f = f[indecies]

    figure = plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(filename)
    plt.clf()


def create_spectrogram(array):
    f, t, Sxx = spectrogram(array, params.sample_rate, window=hann(256, sym=False))
    indecies = np.where((f > params.min_frequency) & (f < params.max_frequency))
    Sxx = Sxx[indecies]
    f = f[indecies]

    figure = plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig("temp.jpg")
    plt.clf()

    time.sleep(0.1)

    rgb = cv2.imread("temp.jpg")
    rgb = rgb[59:428, 80:579]
    
    rgb = cv2.resize(rgb, (32, 5))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
     

    return grayscale

def find_first_chirp(arr, debug_spec=False):
    # Scan at most the first interval for the first chirp
    sliced_arr = arr[:int(params.interval_samples)]

    if debug_spec:
        debug_spectrogram(sliced_arr, "offset.jpg")

    f, t, Sxx = spectrogram(sliced_arr, params.sample_rate, window=hann(256, sym=False))
    # Only handle high frequencies
    high_frequency_indices = np.where((f > params.min_frequency) & (f < params.max_frequency))
    Sxx = Sxx[high_frequency_indices]

    # Calculate the highest point of intensity to find the chirp
    end_of_chirps = np.argmax(Sxx, axis=1)

    counts = np.bincount(end_of_chirps)
    chirp_cut_off = np.argmax(counts)
    time_of_cut_off = t[chirp_cut_off]

    return int(time_of_cut_off * params.sample_rate)