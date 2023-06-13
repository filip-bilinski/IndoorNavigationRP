import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


class CNN_clasifier():

    def __init__(self):
        self.model = None
        self.trained = False
        self.initialized = False
        

    def create_new_model(self, number_of_rooms):
        model = models.Sequential()

        model.add(
            layers.Conv2D(
                16,
                (4, 4),
                activation='relu',
                input_shape=(5,32, 1),
                strides=(1, 1),
                padding="same"
            )
        )
        model.add(
            layers.MaxPooling2D(
                    (
                        2,
                        2,
                    ),
                    strides=(2, 2),
                    padding="valid"
            )
        )
        

        model.add(
            layers.Conv2D(
                32,
                (4, 4),
                activation='relu',
                padding='same'
            )
        )
        model.add(
            layers.MaxPooling2D(
                (2,2)
            )
        )

        model.add(layers.Flatten())

        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(number_of_rooms, activation='softmax'))

        model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        self.model = model
        self.trained = False
        self.initialized = True
        self.number_of_rooms = number_of_rooms
    
    def tarin_model(self, training_data, training_labels, epochs=50, validation_data=None):
        if self.trained:
            print("Model already tarined, initialize new model first")
            return

        if not self.initialized:
            print("Model not initialized")
            return

        if validation_data is not None:
            validation_images, validation_labels = validation_data
            validation_images = np.asarray(validation_images)
            validation_labels = np.asarray(validation_labels)
            validation_data = (validation_images, tf.keras.utils.to_categorical(validation_labels, self.number_of_rooms))

        training_data = np.asarray(training_data)
        training_labels = np.asarray(training_labels)
        training_labels = tf.keras.utils.to_categorical(training_labels, self.number_of_rooms)


        history = self.model.fit(training_data, training_labels, batch_size=1, epochs=epochs, validation_data=validation_data)
        self.trained = True
        if validation_data is not None:
            self.evaluate_model(history, validation_data[0], validation_data[1])

    def evaluate_model(self, history, test_images, test_labels):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig("training_report.jpg")

        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
        

    def save_model(self, filename):
        if not self.trained:
            print("Model has not been trained yet")
            return
        
        self.model.save('./models/' + filename)

    def load_model(self, filename):
        self.model = models.load_model('./models/' + filename)
        self.trained = True
        self.initialized = True

    def summary(self):
        if self.model is None:
            print("Model not initialized yet")
            return

        self.model.summary()

    def run(self, grayscale):
        if not self.trained:
            return "No model trained"

        label = self.model.predict(np.array( [grayscale,]))
        return label


if __name__ == '__main__':
    clasifier = CNN_clasifier()
    clasifier.create_new_model(10)
    clasifier.summary()