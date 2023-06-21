import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model

import cv2

from util import load_images_folder

class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(5, 32, 1)),
            layers.Conv2D(16, (5, 5), activation='tanh', padding='same', strides=1),
            layers.MaxPooling2D((2,2,),strides=(1, 1),padding="same"),
            layers.Conv2D(16, (5, 5), activation='tanh', padding='same', strides=1),
            layers.MaxPooling2D((2,2),strides=(1, 1),padding="same")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(16, kernel_size=5, strides=1, activation='tanh', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=5, strides=1, activation='tanh', padding='same'),
            layers.Conv2D(1, kernel_size=(5, 5), activation='sigmoid', padding='same')
        ])

        
        
        self.decoder.build(input_shape=(None, 5, 32, 16))

        self.compile(optimizer='adam', loss=losses.MeanSquaredError())
        
        self.trained = False

        self.encoder.summary()
        self.decoder.summary()

    
    def train(self, noisy_images_train, images_train, validation_images=None, training_report=False):
        history = self.fit(noisy_images_train, images_train, epochs=50, shuffle=True, validation_data=validation_images, batch_size=1)
        self.trained = True

        if training_report:
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label = 'val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim([0.0, 0.02])
            plt.legend(loc='lower right')
            plt.savefig("training_report.jpg")

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


def main():
    images_noise = np.asarray(load_images_folder('autoencoder_data/with_noise'))
    images = np.asarray(load_images_folder('autoencoder_data/no_noise'))

    images_noise_train, images_noise_test, images_train, images_test = train_test_split(images_noise, images, test_size=(1/5), random_state=42)


    autoencoder = Autoencoder()
    # autoencoder.train(images_noise_train, images_train, validation_images=(images_noise_test, images_test), training_report=True)
    autoencoder.load_model()

    example_im = cv2.imread("autoencoder_data/with_noise/bedroom1687087531.3937702.jpg", cv2.IMREAD_GRAYSCALE)
    spectrogram = np.array(example_im, dtype=np.float32) / 255.0
    example_output = (autoencoder.call(np.array([spectrogram, ])).numpy()[0].reshape((5, 32)) * 255.0).astype(np.uint8)

    cv2.imwrite("example_input.jpg", example_im)
    cv2.imwrite("example_output.jpg", example_output)
    
    # autoencoder.save_model()


if __name__ == "__main__":
    main()