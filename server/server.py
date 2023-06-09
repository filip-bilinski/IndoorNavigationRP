from flask import Flask, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from db_handler import db_handler
import pymongo
from clasifier import CNN_clasifier
from sklearn.model_selection import train_test_split
import util
import time
import cv2
APP = Flask(__name__)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = db_handler(client, "RP")
clasifier = CNN_clasifier()

params = util.globals()
debug = True


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

    np_arr = np_arr[0, int(3 * params.interval_samples): int((params.chirp_amount - 3) * params.interval_samples)]
    offset, duration = util.cross_corelation(np_arr[:int(params.interval_samples)])

    for i in range(params.chirp_amount - 7):
        
        # Slice the array with the offset so that chirp is at the begining of the slice
        offset, duration = util.cross_corelation(np_arr[int(i * params.interval_samples):int((i + 1) * params.interval_samples)])
        start_rate = int(i * params.interval_samples + offset + duration / 2)
        if offset == 0 and duration == 0:
            continue
        
        offset, duration = util.cross_corelation(np_arr[int(start_rate):int(start_rate + params.interval_samples)])
        end_rate = int(start_rate + offset - duration / 2)
        if offset == 0 and duration == 0:
            continue

        sliced = np_arr[start_rate:end_rate]
        
        
        # Create spectrogram
        spectrogram = util.create_spectrogram(sliced)
        # cv2.imwrite("autoencoder_data/bedroom" + str(time.time()) + ".jpg", spectrogram)

        if debug and i < 20:
            cv2.imwrite('spectrogram' + str(counter) + '.jpg', spectrogram)
        counter += 1


        # Save entry to database
        data = {
            u'building': building_label,
            u'room': room_label,
            u'audio': spectrogram.tolist()
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
    images_train, images_test, labels_train, labels_test = train_test_split(data, labels, test_size=(1/6), random_state=42)
    print(len(labels), len(data))
    clasifier.tarin_model(images_train, labels_train, validation_data=(images_test, labels_test))

    return 'OK'
    

@APP.route('/clasify', methods=['POST'])
def calsify_room():
    room_data = request.json
    room_audio = room_data['audio']
    

    np_arr = np.asarray(room_audio, dtype=np.int16)
    np_arr = np_arr[0, int(3 * params.interval_samples):]

    offset, duration = util.cross_corelation(np_arr[:int(params.interval_samples)], "cross_corealtion.jpg")

    start_rate = int(offset + duration / 2)
    offset, duration = util.cross_corelation(np_arr[start_rate:start_rate + int(params.interval_samples)], "cross_corealtion.jpg")
    end_rate = int(start_rate + offset - duration / 2)

    print("True lenght: ", (end_rate - start_rate) / params.sample_rate)


    np_arr = np_arr[start_rate:end_rate]
    util.debug_spectrogram(np_arr, "clasificaion.jpg")
    grayscale = util.create_spectrogram(np_arr)


    prediction = clasifier.run(grayscale)
    print(prediction)
    int_label = np.argmax(prediction[0])
    label = db.int_label_to_room(int_label)

    return label

if __name__ == '__main__':
    APP.run(host='0.0.0.0', debug=True)

