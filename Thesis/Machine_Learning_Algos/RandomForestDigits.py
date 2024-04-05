from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets

digits_data = datasets.load_digits()

image_features = digits_data.images.reshape((len(digits_data.images), -1))  # makes 1D arrays of 1D arrays. Flattens out array
image_targets = digits_data.target

random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt')  # n_jobs = -1 uses all cores for parallel processing

feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=0.3)

# Parameter tuning by finding the best combination of the parameters
param_grid = {
    "n_estimators": [10, 100, 500, 1000],
    "max_depth": [1, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 10, 15, 30, 50]
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
grid_search.fit(feature_train, target_train)
print(grid_search.best_params_)

optimal_estimators = grid_search.best_params_.get('n_estimators')
optimal_depth = grid_search.best_params_.get('max_depth')
optimal_leaf = grid_search.best_params_.get('min_samples_leaf')

print("Optimal n_estimators: %s" % optimal_estimators)
print("Optimal optimal_depth: %s" % optimal_depth)
print("Optimal optimal_leaf: %s" % optimal_leaf)

grid_predictions = grid_search.predict(feature_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))