import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_SET_PATH = 'GenreClassification.json'


# load data
def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


if __name__ == "__main__":

    # load data
    inputs, targets = load_data(DATA_SET_PATH)

    # split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    # Build architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd Hidden layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd hidden layers
        keras.layers.Dense(65, activation='relu'),

        # Output layers
        keras.layers.Dense(9, activation='softmax')

    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

     # Train network
    model.fit(inputs_train, targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=100,
              batch_size=32)

