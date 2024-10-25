# ref: https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20regression.ipynb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, 
                 right=None, var_red=None, value=None) -> None:
        # for decision tree
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None  # root of the tree
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        pass

    def build_tree(self, dataset, curr_depth=0):
        X, y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split['var_red'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree,
                            right_subtree, best_split['var_red'])
        leaf_value = np.mean(y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -float("inf")
        # go through all unique values for every feature,make split and calculate variance reduction
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
                    # higher variance means better split
                    curr_var_red = self.variance_reduction(target, left_y, right_y)
                    if curr_var_red > max_var_red:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['var_red'] = curr_var_red
                        max_var_red = curr_var_red
        return best_split

    def split(self, dataset, feature_index, threshold):
        database_left = np.array(
            [row for row in dataset if row[feature_index] <= threshold]
        )
        database_right = np.array(
            [row for row in dataset if row[feature_index] > threshold]
        )
        return database_left, database_right
    
    def variance_reduction(self,y, left_y, right_y):
        weight_l = len(left_y) / len(y)
        weight_r = len(right_y) / len(y)
        # reduction = total variance before split - variance after split
        reduction = np.var(y) - (weight_l * np.var(left_y) + weight_r * np.var(right_y))
        return reduction

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
model.fit(train_X, train_y)
# dataset = np.concatenate((train_X, train_y), axis=1)
# model.build_tree(dataset)

# print(np.concatenate((train_X, train_y), axis=1))
