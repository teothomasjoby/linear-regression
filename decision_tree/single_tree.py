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
            best_split = self.get_best_split()


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
print(len(train_X), train_y)
