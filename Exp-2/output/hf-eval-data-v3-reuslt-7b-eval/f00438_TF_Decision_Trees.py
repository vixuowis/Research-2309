# function_import --------------------

import tensorflow as tf

# function_code --------------------

class TF_Decision_Trees:
    def __init__(self, input_features, target_threshold):
        self.input_features = input_features
        self.target_threshold = target_threshold
        self.model = None

    def fit(self, dataset):
        # Implement the model training here
        pass

    def predict(self, input_features):
        # Implement the prediction here
        return [0]

"""
# main ----------------------------
"""
tf.reset_default_graph()

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Split the iris dataset into features and target (class)
X = pd.DataFrame(iris["data"])
y = pd.Series(iris["target"])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree Model --------------------
decision_trees = TF_Decision_Trees(input_features=X, target_threshold=y)
decision_trees.fit(dataset=(X_train, X_test))
prediction = decision_trees.predict(input_features=X_test)

# test_function_code --------------------

def test_TF_Decision_Trees():
    input_features = {'age': 30, 'workclass': 'Private', 'education': 'Bachelors', 'marital_status': 'Never-married',
                   'occupation': 'Tech-support', 'relationship': 'Not-in-family', 'race': 'White',
                   'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week': 40,
                   'native_country': 'United-States'}
    model = TF_Decision_Trees(input_features, target_threshold=50_000)
    assert model.input_features == input_features
    assert model.target_threshold == 50_000
    assert model.predict(input_features) == [0]
    return 'All Tests Passed'


# call_test_function_code --------------------

test_TF_Decision_Trees()