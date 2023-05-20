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

APP = Flask(__name__)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = db_handler(client, "RP")

min_frequency = 19500
max_frequency = 20500

interval = 0.1
sample_rate = 44100
chirp_amount = 504

interval_rate = sample_rate * interval

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
    
    return rgb

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
    rooms = db.retrieve_rooms()
    
    return rooms

@APP.route('/add_room', methods=['POST'])
def add_room():

    room_data = request.json
    room_label = room_data['room_label']
    building_label = room_data['building_label']
    room_audio = room_data['audio']
    
    
    

    counter = 0
    np_arr = np.asarray(room_audio, dtype=np.int16)

    #create_spectrogram(np_arr[0], "whole.jpg")

    for i in range(2, chirp_amount - 2):

        start_rate = int(i * interval_rate)
        sliced = np_arr[0,start_rate:(int(start_rate + interval_rate))]
        
        print(np.argmax(sliced))
        start_rate = int(i * interval_rate + np.argmax(sliced))
        sliced = np_arr[0,start_rate:(int(start_rate + interval_rate))]
        
        if i < 20:
            debug_spectrogram(sliced, 'tarck_cut' + str(counter) + '.jpg', 5)
            debug_spectrogram(sliced, 'tarck' + str(counter) + '.jpg', 0)
        counter += 1

        rgb = create_spectrogram(sliced, 5)

        data = {
        u'building': building_label,
        u'room': room_label,
        u'audio': rgb.tolist()
        }

        db.add_entry(building_label, data)


    return 'OK'


@APP.route('/clasify', methods=['POST'])
def calsify_room():
    audio = request.files['audio']
    #TODO: run the clasifier
    result = 'room_1'
    return result

if __name__ == '__main__':
    APP.run(host='0.0.0.0', debug=True)

