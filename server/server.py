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

APP = Flask(__name__)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = db_handler(client, "RP")
clasifier = CNN_clasifier()

min_frequency = 19500
max_frequency = 20500

interval = 0.1
sample_rate = 44100
chirp_amount = 10

interval_rate = sample_rate * interval

def find_first_chirp(arr):
    # Scan at most the first interval for the first chirp
    sliced_arr = arr[0, int(2*interval_rate):int(3*interval_rate)]
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


def create_spectrogram(array, cut_offset):
    f, t, Sxx = spectrogram(array, 44100, window=hann(256, sym=False))

    
    indecies = np.where((f > min_frequency) & (f < max_frequency))

    
    time_segments = t.shape[0] - cut_offset
    Sxx = Sxx[indecies, cut_offset:time_segments]
    Sxx = Sxx[0]
    f = f[indecies]
    t = t[cut_offset:time_segments]

    figure = plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig("temp.jpg")
    plt.clf()

    time.sleep(0.2)

    rgb = cv2.imread("temp.jpg")
    rgb = rgb[59:428, 80:579]
    rgb = cv2.resize(rgb, (32, 5))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    return grayscale

def debug_spectrogram(array, filename, cut_offset):
    f, t, Sxx = spectrogram(array, 44100, window=hann(256, sym=False))

    
    indecies = np.where((f > min_frequency) & (f < max_frequency))

    
    time_segments = t.shape[0] - cut_offset
    Sxx = Sxx[indecies, cut_offset:time_segments]
    Sxx = Sxx[0]
    f = f[indecies]
    t = t[cut_offset:time_segments]

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
    offset = find_first_chirp(np_arr)

    for i in range(2, chirp_amount - 2):
        
        # Slice the array with the offset so that chirp is at the begining of the slice
        start_rate = int(i * interval_rate + offset)
        sliced = np_arr[0,start_rate:(int(start_rate + interval_rate))]
        
        if i < 20:
            debug_spectrogram(sliced, 'tarck_cut' + str(counter) + '.jpg', 5)
            debug_spectrogram(sliced, 'tarck' + str(counter) + '.jpg', 0)
        counter += 1

        # Create spectrogram
        rgb = create_spectrogram(sliced, 5)

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
    #TODO

    return 'OK'

@APP.route('/clasify', methods=['POST'])
def calsify_room():
    audio = request.files['audio']
    #TODO: run the clasifier
    result = 'room_1'
    return result

if __name__ == '__main__':
    APP.run(host='0.0.0.0', debug=True)

