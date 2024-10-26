# ref: https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, 
                 right=None, info_gain=None, value=None) -> None:
        # for decision tree
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
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
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree,
                            right_subtree, best_split['info_gain'])
        leaf_value = np.mean(y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")
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
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def split(self, dataset, feature_index, threshold):
        database_left = np.array(
            [row for row in dataset if row[feature_index] <= threshold]
        )
        database_right = np.array(
            [row for row in dataset if row[feature_index] > threshold]
        )
        return database_left, database_right
    
    def information_gain(self,y, left_y, right_y, mode="entropy"):
        weight_l = len(left_y) / len(y)
        weight_r = len(right_y) / len(y)
        # gain = total entropy before split - entropy after split
        if mode == "gini":
            gain = self.gini_index(y) - (weight_l*self.gini_index(left_y) + weight_r*self.gini_index(right_y))
        else:
            gain = self.entropy(y) - (weight_l*self.entropy(left_y) + weight_r*self.entropy(right_y))

        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls**2
        return 1-gini

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
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

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
model = DecisionTreeClassifier(2, 10)
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
model.print_tree()
print("accuracy = ", accuracy_score(test_y, y_pred))
