from clasifier import CNN_clasifier
from autoencoder import Autoencoder

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def create_eval_dataset(path):
    images = []
    labels = []
    
    files = os.listdir(path)
    files.sort()

    for file in files:
        
        if file.startswith('bedroom'):
            labels.append(3)
        elif file.startswith('laundry'):
            labels.append(0)
        elif file.startswith('bathroom'):
            labels.append(1)
        else:
            labels.append(2)

        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        images.append(img)

    
    return images, labels

def run_test(labels, images_no_ae, images_ae, cnn_clasifier, ae_cnn_clasifier, filename):
    no_ae_predictions = []
    ae_predictions = []

    corr_ae = 0
    corr = 0

    for i in range(min(len(images_no_ae), len(images_ae))):

        prediction = cnn_clasifier.run(images_no_ae[i])
        label = np.argmax(prediction[0])
        no_ae_predictions.append(label)

        if label == labels[i]:
            corr +=1

        prediction = ae_cnn_clasifier.run(images_ae[i])
        label = np.argmax(prediction[0])
        ae_predictions.append(label)

        if label == labels[i]:
            corr_ae +=1

    conf_matrix_no_ae = confusion_matrix(labels[:len(no_ae_predictions)], no_ae_predictions)
    conf_matrix_ae = confusion_matrix(labels[:len(ae_predictions)], ae_predictions)

    conf_disp = ConfusionMatrixDisplay(conf_matrix_ae)
    conf_disp.plot()
    plt.savefig(filename + '_ae.jpg')
    plt.clf()

    conf_disp = ConfusionMatrixDisplay(conf_matrix_no_ae)
    conf_disp.plot()
    plt.savefig(filename + '_no_ae.jpg')
    plt.clf()

    return corr/ len(labels), corr_ae / len(labels)

def main():
    images_no_music, labels_no_music = create_eval_dataset('experiment_data/no_noise')
    images_with_music, labels_with_music = create_eval_dataset('experiment_data/with_noise')

    images_no_music_ae = []
    images_with_music_ae = []

    ae = Autoencoder()
    ae.load_model()

    for i in range(min(len(images_with_music), len(images_no_music))):

        spectrogram = images_no_music[i].astype(np.float32) / 255.0
        spectrogram = (ae.call(np.array([spectrogram, ])).numpy()[0].reshape((5, 32)) * 255.0).astype(np.uint8)

        images_no_music_ae.append(spectrogram)

        spectrogram = images_with_music[i].astype(np.float32) / 255.0
        spectrogram = (ae.call(np.array([spectrogram, ])).numpy()[0].reshape((5, 32)) * 255.0).astype(np.uint8)

        images_with_music_ae.append(spectrogram)
        


    cnn_clasifier = CNN_clasifier()
    cnn_clasifier.load_model('model')
    ae_cnn_clasifier = CNN_clasifier()
    ae_cnn_clasifier.load_model('model+ae')

    acc_n_n, acc_y_n = run_test(labels_no_music, images_no_music, images_no_music_ae, cnn_clasifier, ae_cnn_clasifier, 'no_music')
    acc_n_y, acc_y_y = run_test(labels_with_music, images_with_music, images_with_music_ae, cnn_clasifier, ae_cnn_clasifier, 'with_music')

    mixed_labels = labels_no_music + labels_with_music
    mixed_images = images_no_music + images_with_music
    mixed_images_ae = images_no_music_ae + images_with_music_ae

    acc_mixed_n, acc_mixed_y = run_test(mixed_labels, mixed_images, mixed_images_ae, cnn_clasifier, ae_cnn_clasifier, 'mixed')


    print('Accuracy no music no autoencoder: ', acc_n_n)
    print('Accuracy no music with autoencoder: ', acc_y_n)
    print('Accuracy with music no autoencoder: ', acc_n_y)
    print('Accuracy with music with autoencoder: ', acc_y_y)
    print('Accuarcy mixed no autoencoder: ', acc_mixed_n)
    print('Accuarcy mixed with autoencoder: ', acc_mixed_y)
    
    
    
if __name__ == "__main__":
    main()
