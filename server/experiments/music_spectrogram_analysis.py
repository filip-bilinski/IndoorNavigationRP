import numpy as np
import cv2
from scipy.io import wavfile
from util import globals, cross_corelation, create_spectrogram

def analyze_spectrograms(spectrograms):

    spectrograms = np.array(spectrograms)

    means = []

    for spectrogram in spectrograms:
        means.append(np.mean(spectrogram))

    print("Spectrogram mean:", np.mean(means), np.std(means))

    mean_spectrogram = np.mean(spectrograms, axis=0)
    mean_spectrogram = cv2.resize(mean_spectrogram, (mean_spectrogram.shape[1] * 10, mean_spectrogram.shape[0] * 10))
    cv2.imwrite("mean_spectrogram.jpg", mean_spectrogram)
    return


def run_music_experiment(room_audio):
    params = globals()
    np_arr = np.asarray(room_audio, dtype=np.int16)

    np_arr = np_arr[0, int(3 * params.interval_samples): int((params.chirp_amount - 3) * params.interval_samples)]
    offset, duration = cross_corelation(np_arr[:int(params.interval_samples)])

    spectrograms_for_analysis = []

    for i in range(params.chirp_amount - 7):
        
        # Slice the array with the offset so that chirp is at the begining of the slice
        offset, duration = cross_corelation(np_arr[int(i * params.interval_samples):int((i + 1) * params.interval_samples)])
        start_rate = int(i * params.interval_samples + offset + duration / 2)
        if offset == 0 and duration == 0:
            continue
        
        offset, duration = cross_corelation(np_arr[int(start_rate):int(start_rate + params.interval_samples)])
        end_rate = int(start_rate + offset - duration / 2)
        if offset == 0 and duration == 0:
            continue

        sliced = np_arr[start_rate:end_rate]
        
        # Create spectrogram
        spectrogram = create_spectrogram(sliced)

        spectrograms_for_analysis.append(spectrogram)

    analyze_spectrograms(spectrograms_for_analysis)

    return


def main():
    fs, audio_data = wavfile.read('audio.wav')
    
    run_music_experiment([audio_data])

if __name__ == "__main__":
    main()

