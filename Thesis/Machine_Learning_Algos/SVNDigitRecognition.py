import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:

data = digits.images.reshape(len(digits.images), -1)

classifier = svm.SVC(gamma=0.001)

# 75% of original data set is for training
train_test_split = int(len(digits.images) * 0.75)
classifier.fit(data[:train_test_split], digits.target[:train_test_split])

# now predict the value of teh digits on the 25%
expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

# lets test on last few images
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for test image: ", classifier.predict(data[-3].reshape(1,-1)))
plt.show()

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))