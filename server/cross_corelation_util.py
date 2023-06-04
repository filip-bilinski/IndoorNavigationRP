import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

sample_rate = 44100
frequency = 20000
duration = 0.02 

def cross_corelation(arr):

    size = len(arr)
    samples = (np.sin(2 * np.pi * np.arange(duration * sample_rate) * frequency / sample_rate)).astype(np.float32)

    corr = signal.correlate(arr, samples, mode='valid', method='fft')
    size = len(corr)
    plt.plot(np.arange(size) / sample_rate, corr)
    plt.savefig("cross_corealtion.jpg")
    plt.clf()


    print("MAX", np.max(corr),"ARGMAX", np.argmax(corr))
    return np.argmax(corr)