import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

import util

params = util.globals()

def cross_corelation(arr, filename=None):

    samples = (np.sin(2 * np.pi * np.arange(params.duration * params.sample_rate) * params.frequency / params.sample_rate)).astype(np.float32)

    corr = signal.correlate(arr, samples, mode='same', method='fft')
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


if __name__ == "__main__":
    samplerate, data = wavfile.read('audio.wav')
    np_arr = np.asarray(data, dtype=np.int16)

    np_arr = np_arr[0, int(3 * params.interval_samples):]

    offset = util.find_first_chirp(np_arr, debug_spec=True)
    print("chirp loc: ", offset)
    cor_max = cross_corelation(np_arr[:int(params.interval_samples)], "lul.jpg")


