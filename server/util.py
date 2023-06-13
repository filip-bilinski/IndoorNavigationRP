from scipy.signal import spectrogram, correlate
from scipy.signal.windows import hann
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import os
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
        self.chirp_amount = 107
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

    figure = plt.pcolormesh(t, f, Sxx, shading='nearest')
    plt.axis('off')
    plt.savefig("temp.jpg", bbox_inches='tight', pad_inches=0)
    plt.clf()

    time.sleep(0.005)

    rgb = cv2.imread("temp.jpg")
    
    rgb = cv2.resize(rgb, (32, 5))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    ratio = 255/np.max(grayscale)
    grayscale = (grayscale * ratio).astype(np.uint8)

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

def cross_corelation(arr, filename=None):

    samples = (np.sin(2 * np.pi * np.arange(params.duration * params.sample_rate) * params.frequency / params.sample_rate)).astype(np.float32)

    corr = correlate(arr, samples, mode='same', method='fft')
    size = len(corr)

    started = False
    chirp_start = []
    chirp_end = []
    for i in range(size):
        if abs(corr[i]) > 100000 and not started:
            started = True
            chirp_start.append(i)
        if abs(corr[i]) < 100000 and started:
            chirp_end.append(i)
            started = False

    if len(chirp_start) == 0 or len(chirp_end) == 0:
        return 0, 0

    
    calculated_chirp_duration = (chirp_end[len(chirp_end) - 1] - chirp_start[0])
    chirp_midle = chirp_start[0] + calculated_chirp_duration / 2

    if filename is not None:
        plt.plot(np.arange(size) / params.sample_rate, corr)
        plt.savefig(filename)
        plt.clf()   


    return chirp_midle, calculated_chirp_duration


def load_images_folder(path):
    images = []
    
    files = os.listdir(path)
    files.sort()

    for file in files:
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    
    return images


if __name__ == "__main__":
    samplerate, data = wavfile.read('audio.wav')
    np_arr = np.asarray(data, dtype=np.int16)

    np_arr = np_arr[0, int(3 * params.interval_samples):]

    offset = util.find_first_chirp(np_arr, debug_spec=True)
    print("chirp loc: ", offset)
    cor_max = cross_corelation(np_arr[:int(params.interval_samples)], "lul.jpg")
