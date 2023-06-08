import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import os
import cv2

class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(5, 32, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

        
        
        self.decoder.build(input_shape=(None, 2, 8, 8))

        self.compile(optimizer='adam', loss=losses.MeanSquaredError())
        
        self.trained = False

        self.encoder.summary()
        self.decoder.summary()

    
    def train(self, noisy_images_train, images_train, validation_images=None):
        self.fit(noisy_images_train, images_train, epochs=100, shuffle=True, validation_data=validation_images, batch_size=1)
        self.trained = True

    def call(self, image):
        encoded = self.encoder(image)
        return self.decoder(encoded)


def load_images_folder(path):
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    
    return images

def main():
    images_noise = np.asarray(load_images_folder('autoencoder_data/with_noise'))
    images = np.asarray(load_images_folder('autoencoder_data/no_noise'))
    print(images_noise.shape, images.shape)

    images_noise_train, images_noise_test, images_train, images_test = train_test_split(images_noise, images, test_size=0.1, random_state=42)


    autoencoder = Autoencoder()
    autoencoder.train(images_noise_train, images_train, validation_images=(images_noise_test, images_test))

    example_im = cv2.imread("autoencoder_data/with_noise/laundry1686235109.884637.jpg", cv2.IMREAD_GRAYSCALE)
    print(example_im.shape)
    example_output = autoencoder.call(np.array([example_im,])).numpy()[0].reshape((5, 32))

    print(np.max(example_output), np.min(example_output))
    print(example_output)
    print(example_output.shape, example_im.shape, example_output.dtype, example_im.dtype)

    combined_im = cv2.vconcat([example_im, example_output])
    cv2.imwrite("denoise_output.jpg", combined_im)


if __name__ == "__main__":
    main()