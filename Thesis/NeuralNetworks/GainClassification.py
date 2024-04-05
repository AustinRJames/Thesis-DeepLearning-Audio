import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


gain_data = pd.read_csv('../CSVs/GainChange.csv')
targets = pd.read_csv('../CSVs/GainChangeLabels.csv')

features = np.array(gain_data).reshape(-1, 480)  # turn into array not dataframe
y = np.array(targets).reshape(-1, 1)  # 0 for gain loss, 1 for added gain

# # First class: (1,0), second class: (0, 1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(10, input_dim=480, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=10000, verbose=2)

results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print("Predictions after the training...:")
print(model.predict(test_features))

results = model.evaluate(test_features, test_targets)
print("Accuracy on test dataset: %.2f" % results[1])