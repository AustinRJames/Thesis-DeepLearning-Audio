from sklearn.datasets import fetch_olivetti_faces
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti_data = fetch_olivetti_faces()

# there are 400 images - 10x40 ( 40 people - 1 person has 10 images) - 1 image = 64x64 pixels
features = olivetti_data.data
# we represent target variables(people) with integers (faced ids)
targets = olivetti_data.target

# split the origina data-set (training and test set)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)

# let's try to find the optimal number of eigenvectors(principle component)
pca = PCA(n_components=100, whiten=True)
pca.fit(X_train)
X_pca = pca.fit_transform(features)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# let's use the machine learning models

models = [('Logistic Regression', LogisticRegression()), ("Support Vector Machine", SVC()),
          ("Naive Bayes", GaussianNB())]

for name, model in models:

    # Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, X_pca, targets, cv=kfold)
    print("Mean of the cross-validation scores: %s" % cv_scores.mean())





# Plotting faces
# See only one face but all of their pictures
# fig, sub_plots = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))
# for j in range(10):
#     sub_plots[j].imshow(features[j].reshape(64, 64), cmap='gray')
#     sub_plots[j].set_xticks([])
#     sub_plots[j].set_yticks([])
#     sub_plots[j].set_title("Face id = 0" )

# plt.show()

# How to see all of the faces
# fig, sub_plots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
# sub_plots = sub_plots.flatten()
#
# for unique_user_id in np.unique(targets):
#     image_index = unique_user_id * 8
#     sub_plots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap='gray')
#     sub_plots[unique_user_id].set_xticks([])
#     sub_plots[unique_user_id].set_yticks([])
#     sub_plots[unique_user_id].set_title("Face id: %s" % unique_user_id)
#
# plt.suptitle("The dataset (40 people")
# plt.show()
