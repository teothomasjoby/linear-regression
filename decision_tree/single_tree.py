# ref: https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20regression.ipynb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None  # root of the tree
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, X, y, curr_depth=0):
        num_samples, num_features = np.shape(X)
        best_split = {}

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(X, y, num_samples, num_features)

    def get_best_split(self, X, y, num_samples, num_features):
        dataset = np.concatenate((X, y), axis=1)
        best_split = {}

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_threshold = np.unique(feature_values)
            for threshold in possible_threshold:
                dataset_left, dataset_right = self.split(
                    dataset, feature_index, threshold
                )
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    target, left_y, right_y = (
                        dataset[:, -1],
                        dataset_left[:, -1],
                        dataset_right[:, -1],
                    )
                    curr_var_red = self.variance_reduction(target, left_y, right_y)
            print("feature_values", feature_values)
            print("possible_threshold", possible_threshold)

    def split(self, dataset, feature_index, threshold):
        database_left = np.array(
            [row for row in dataset if row[feature_index] <= threshold]
        )
        database_right = np.array(
            [row for row in dataset if row[feature_index] > threshold]
        )
        return database_left, database_right


col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "type"]
data = pd.read_csv("data/Iris.csv", skiprows=1, header=None, names=col_names)

# convert type column to numerical values
data["type"] = data["type"].map(
    {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
)
# train test split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=41
)
model = DecisionTreeClassifier(2, 2)
model.build_tree(train_X, train_y)

print(np.concatenate((train_X, train_y), axis=1))
