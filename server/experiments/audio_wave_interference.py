import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def simulate_interference():
    # Load audio file
    fs, audio_data = wavfile.read('audio.wav')
    audio_data = audio_data.astype(float) / 32767.0  # Normalize audio data to the range [-1, 1]

    # Generate 20kHz signal
    duration = 1000
    t = np.linspace(0, duration, duration)
    signal_20kHz = np.sin(2 * np.pi * 20000 * t)
    audio_data = audio_data[:duration]


    # Perform interference simulation
    interference = audio_data + signal_20kHz

    # Original audio signal
    plt.subplot(3, 1, 1)
    plt.plot(t, audio_data, color='b')
    plt.title('Original Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 20kHz signal
    plt.subplot(3, 1, 2)
    plt.plot(t, signal_20kHz, color='g')
    plt.title('20kHz Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Interference signal
    plt.subplot(3, 1, 3)
    plt.plot(t, interference, color='r')
    plt.title('Interference Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('audio_interference_sim.jpg')

if __name__ == "__main__":
    simulate_interference()
