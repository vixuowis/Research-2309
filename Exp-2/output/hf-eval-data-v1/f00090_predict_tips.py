import dabl
from dabl import SimpleRegressor
from sklearn.datasets import load_boston

# Function to predict tips
# This function uses a pre-trained model 'merve/tips9y0jvt5q-tip-regression' to predict the amount of tips
# The model is trained on a dataset of tips and uses Ridge regression with alpha set to 10
# The function takes as input a dataframe with the same structure as the training data
# It returns a series of predicted tip amounts

def predict_tips(data):
    target_column = 'tip'
    regressor = SimpleRegressor()
    model = regressor.fit(data, target=target_column)
    predicted_tips = model.predict(data)
    return predicted_tips