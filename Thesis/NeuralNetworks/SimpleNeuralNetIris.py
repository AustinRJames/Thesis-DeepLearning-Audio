from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.optimizers import Adam

iris_data = load_iris()

features = iris_data.data
labels = iris_data.target.reshape(-1, 1)

# We have 3 classes so the labels will have 3 values:
# first class: (1, 0, 0) , second class: (0, 1, 0), third class: (0, 0, 1)
encoder = OneHotEncoder()
target = encoder.fit_transform(labels).toarray()

train_features, test_features, train_labels, test_labels = train_test_split(features, target, test_size=0.3)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for weights: SGD, Adagrad, ADAM
optimizer = (Adam(learning_rate=0.001))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=10000, batch_size=20, verbose=2)
results = model.evaluate(test_features, test_labels, use_multiprocessing=True)

print("Predictions after the training...:")
print(model.predict(test_features))
