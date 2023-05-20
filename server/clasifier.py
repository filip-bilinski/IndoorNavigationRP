import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class CNN_clasifier():

    def __init__(self, number_of_rooms):

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
        model.add(layers.Dense(number_of_rooms, activation='relu'))

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.model = model



    def summary(self):
        self.model.summary()


if __name__ == '__main__':
    clasifier = CNN_clasifier(10)
    clasifier.summary()