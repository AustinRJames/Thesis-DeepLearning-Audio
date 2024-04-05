import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_SET_PATH = 'GenreClassification.json'


# load data
def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='test accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title("Accuracy Eval")

    axs[1].plot(history.history['loss'], label='train error')
    axs[1].plot(history.history['val_loss'], label='test error')
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].set_title("Error Eval")

    plt.show()


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
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.1),

        # 2nd Hidden layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.1),

        # 3rd hidden layers
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.1),

        # Output layers
        keras.layers.Dense(10, activation='softmax')

    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    # Train network
    history = model.fit(inputs_train, targets_train,
                        validation_data=(inputs_test, targets_test),
                        epochs=100,
                        batch_size=32)

    # Plot accuracy and error over the epochs
    plot_history(history)
