# ref: https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=3, max_depth=3):
        self.root = None  # root of the tree
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

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
                    y, left_y, right_y = (
                        dataset[:, -1],
                        dataset_left[:, -1],
                        dataset_right[:, -1],
                    )
                    # higher variance means better split
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
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

    def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self. make_prediction(x, tree.right)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

data = pd.read_csv("data/airfoil_noise.csv", skiprows=1, header=None)

# train test split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=41
)
model = DecisionTreeRegressor(2, 2)
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
model.print_tree()
print("error = ", np.sqrt(mean_squared_error(test_y, y_pred)))

