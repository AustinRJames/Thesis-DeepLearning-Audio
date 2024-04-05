import scipy as scipy
import os
import librosa
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as datasets

DATASET_PATH = "../GTZAN_Data/genres_original"
JSON_PATH = "GenreClassification.json"

SAMPLE_RATE = 44100
DURATION = 30  # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
TIME_SEC = np.arange(0, SAMPLES_PER_TRACK) / SAMPLE_RATE



def butter_lowpass(Wn, fs, order=1):
    return scipy.signal.butter(order, Wn, fs=fs, btype='lowpass')


def butter_lowpass_filter(data, Wn, fs, order=2):
    b, a = butter_lowpass(Wn, fs=fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int((SAMPLES_PER_TRACK / num_segments))
    expected_number_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  # want to round up

    # loops through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure that we're not at the root level
        if dirpath is not dataset_path:

            # Save the semantic label
            dirpath_components = dirpath.split("/")  # Genre/blue => ["genre", "blues"]
            semantic_label = dirpath_components[-1]  # only get last component aka genre name
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))


            # Process files for specific genre
            for f in filenames:

                Wn = random.randint(5000, 15000)  # choose random cut off freq from 500-10000 Hz

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, fs = librosa.load(file_path, sr=SAMPLE_RATE)

                signal = butter_lowpass_filter(signal, Wn=Wn,  fs=fs, order=1)  # FILTER SIGNAL

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=fs,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)

                    mfcc = mfcc.T  # Transposing

                    # Store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_number_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())  # Need list to save to JSON file
                        data["labels"].append(Wn - 1)
                        print("{}, segment:{}, Wn:{}".format(file_path, s + 1, Wn))

    with open(json_path, "w") as fp:
        print("\nDUMPING DATA")
        json.dump(data, fp, indent=4)
        print("\n DATA DUMP COMPLETE")


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
