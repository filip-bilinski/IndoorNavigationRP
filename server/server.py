from flask import Flask, request
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from scipy.signal.windows import hann
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from db_handler import db_handler
import pymongo
import cv2
import time
from clasifier import CNN_clasifier
from sklearn.model_selection import train_test_split


APP = Flask(__name__)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = db_handler(client, "RP")
clasifier = CNN_clasifier()

min_frequency = 19500
max_frequency = 20500

interval = 0.1
sample_rate = 44100
chirp_radius = 0.02
interval_samples = sample_rate * interval
chirp_amount = 504

debug = True

interval_rate = sample_rate * interval
chirp_radius_samples = int(sample_rate * chirp_radius/2)

def find_first_chirp(arr):
    # Scan at most the first interval for the first chirp
    sliced_arr = arr[:int(interval_rate)]
    f, t, Sxx = spectrogram(sliced_arr, 44100, window=hann(256, sym=False))
    # Only handle high frequencies
    high_frequency_indices = np.where((f > min_frequency) & (f < max_frequency))
    Sxx = Sxx[high_frequency_indices]

    # Calculate the highest point of intensity to find the chirp
    end_of_chirps = np.argmax(Sxx, axis=1)

    counts = np.bincount(end_of_chirps)
    chirp_cut_off = np.argmax(counts)
    time_of_cut_off = t[chirp_cut_off]

    return int(time_of_cut_off * sample_rate )


def create_spectrogram(array):
    f, t, Sxx = spectrogram(array, 44100, window=hann(256, sym=False))
    indecies = np.where((f > min_frequency) & (f < max_frequency))
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
    
    
    ratio = 255 / np.max(grayscale)
    grayscale = grayscale * ratio
    grayscale = grayscale.astype(np.uint8)

    return grayscale

def debug_spectrogram(array, filename):
    f, t, Sxx = spectrogram(array, 44100, window=hann(256, sym=False))

    
    indecies = np.where((f > min_frequency) & (f < max_frequency))

    
    Sxx = Sxx[indecies]
    f = f[indecies]

    figure = plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(filename)
    plt.clf()

@APP.route('/get_rooms', methods=['GET'])
def get_rooms():
    room_list = []
    response_body = {}
    rooms, _ = db.retrieve_rooms()
    
    return rooms

@APP.route('/add_room', methods=['POST'])
def add_room():

    room_data = request.json
    room_label = room_data['room_label']
    building_label = room_data['building_label']
    room_audio = room_data['audio']

    counter = 0
    np_arr = np.asarray(room_audio, dtype=np.int16)

    np_arr = np_arr[0, int(2 * interval_samples): int((chirp_amount - 2) * interval_samples)]
    offset = find_first_chirp(np_arr)

    for i in range(chirp_amount - 4):
        
        # Slice the array with the offset so that chirp is at the begining of the slice
        start_rate = int(i * interval_samples + offset + chirp_radius_samples)
        end_rate = int((i + 1) * interval_samples + offset - chirp_radius_samples)
        sliced = np_arr[start_rate:end_rate]
        
        if debug and i < 20:
            start_rate_debug = int(i * interval_samples + offset)
            end_rate_debug = int((i + 1) * interval_samples + offset)

            sliced_debug = np_arr[start_rate_debug:end_rate_debug]

            debug_spectrogram(sliced, 'tarck_cut' + str(counter) + '.jpg')
            debug_spectrogram(sliced_debug, 'tarck' + str(counter) + '.jpg')
        counter += 1

        # Create spectrogram
        rgb = create_spectrogram(sliced)

        # Save entry to database
        data = {
        u'building': building_label,
        u'room': room_label,
        u'audio': rgb.tolist()
        }
        db.add_entry(building_label, data)


    return 'OK'

@APP.route('/create_model', methods=['GET'])
def create_model():
    _ , number_of_rooms = db.retrieve_rooms()
    clasifier.create_new_model(number_of_rooms)

    return 'OK'

@APP.route('/save_model', methods=['GET'])
def save_model():
    clasifier.save_model('model')

    return 'OK'

@APP.route('/load_model', methods=['GET'])
def load_model():
    clasifier.load_model('model')

    return 'OK'

@APP.route('/train_model', methods=['GET'])
def train():
    labels, data = db.prepare_training_dataset()
    images_train, images_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42)
    print(len(labels), len(data))
    clasifier.tarin_model(images_train, labels_train, validation_data=(images_test, labels_test))

    return 'OK'
    

@APP.route('/clasify', methods=['POST'])
def calsify_room():
    room_data = request.json
    room_audio = room_data['audio']

    np_arr = np.asarray(room_audio, dtype=np.int16)
    np_arr = np_arr[0, int(2 * interval_samples):]
    offset = find_first_chirp(np_arr)
    start_rate = int(interval_samples + offset + chirp_radius_samples)
    end_rate = int(start_rate + interval_rate)

    np_arr = np_arr[start_rate:end_rate]
    grayscale = create_spectrogram(np_arr)

    prediction = clasifier.run(grayscale)
    print(prediction)
    int_label = np.argmax(prediction[0])
    label = db.int_label_to_room(int_label)

    return label

if __name__ == '__main__':
    APP.run(host='0.0.0.0', debug=True)

