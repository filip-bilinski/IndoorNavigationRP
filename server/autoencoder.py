import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model

import os
import cv2

class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(5, 32, 1)),
            layers.Conv2D(16, (4, 4), activation='relu', padding='same', strides=1),
            layers.MaxPooling2D((2,2), padding='same', strides=1),
            layers.Conv2D(32, (4, 4), activation='relu', padding='same', strides=1),
            layers.MaxPooling2D((2,2), padding='same', strides=1)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(32, kernel_size=4, strides=1, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=4, strides=1, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

        
        
        self.decoder.build(input_shape=(None, 5, 32, 32))

        self.compile(optimizer='adam', loss=losses.MeanSquaredError())
        
        self.trained = False

    
    def train(self, noisy_images_train, images_train, validation_images=None):
        self.fit(noisy_images_train, images_train, epochs=20, shuffle=True, validation_data=validation_images, batch_size=1)
        self.trained = True

    def call(self, image):
        encoded = self.encoder(image)
        return self.decoder(encoded)

    def save_model(self):
        self.encoder.save('models/encoder')
        self.decoder.save('models/decoder')

    def load_model(self):
        self.decoder = load_model('models/decoder')
        self.encoder = load_model('models/encoder')

        self.trained = True



def load_images_folder(path):
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    
    return images

def main():
    images_noise = np.asarray(load_images_folder('autoencoder_data/with_noise'))
    images = np.asarray(load_images_folder('autoencoder_data/no_noise'))

    images_noise_train, images_noise_test, images_train, images_test = train_test_split(images_noise, images, test_size=(1/6), random_state=42)


    autoencoder = Autoencoder()
    # autoencoder.train(images_noise_train, images_train, validation_images=(images_noise_test, images_test))
    autoencoder.load_model()

    example_im = cv2.imread("autoencoder_data/with_noise/bedroom1686294660.6565168.jpg", cv2.IMREAD_GRAYSCALE)
    example_output = autoencoder.call(np.array([example_im,])).numpy()[0].reshape((5, 32))
    example_output = (example_output * 255.0).astype(np.uint8)

    combined_im = cv2.vconcat([example_im, example_output])
    combined_im = cv2.resize(combined_im, (combined_im.shape[1] * 10, combined_im.shape[0] * 10))
    cv2.imwrite("denoise_output.jpg", combined_im)

    # autoencoder.save_model()


if __name__ == "__main__":
    main()