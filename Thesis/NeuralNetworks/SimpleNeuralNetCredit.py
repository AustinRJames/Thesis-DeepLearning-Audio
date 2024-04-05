from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import pandas as pd
import numpy as np

credit_data = pd.read_csv('../CSVs/credit_data.csv')

features = credit_data[["income", "age", "loan"]]
y = np.array(credit_data.default).reshape(-1,1)
print(features)
# We have 2 classes so the labels will have 2 values
# First class: (1,0), second class: (0, 1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.3)

model = Sequential()
model.add(Dense(10, input_dim=3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, verbose=2)

results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print("After Training: ")
print(results)
