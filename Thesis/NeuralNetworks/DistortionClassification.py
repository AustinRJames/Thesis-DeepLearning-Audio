import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

distortion_data = pd.read_csv("../CSVs/DistortionTypes.csv")
y = pd.read_csv('../CSVs/DistortionTypeLabels.csv')

features = np.array(distortion_data).reshape(-1, 480)
y = np.array(y).reshape(-1,1)

# Cubic Distortion (1,0, 0), Arc tan Distortion: (0, 1, 0), Half-wave Rect: (0, 0, 1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.25, random_state=0)

model = Sequential()

# first parameter is output dimension
model.add(Dense(64, input_dim=480, activation='relu'))
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for weights: SGD, Adagrad, ADAM
optimizer = (Adam(learning_rate=0.005))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=500, batch_size=20, verbose=2)

results = model.evaluate(test_features, test_targets)
print("Accuracy on test dataset: %.2f" % results[1])