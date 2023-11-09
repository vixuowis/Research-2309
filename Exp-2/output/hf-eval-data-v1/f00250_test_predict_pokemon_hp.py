import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
pokemon_data = pd.read_csv('julien-c/kaggle-rounakbanik-pokemon.csv')
# Split the dataset into training and testing sets
train_data, test_data = train_test_split(pokemon_data, test_size=0.2)

# Select a sample from the test data
sample_data = test_data.sample()
# Prepare the input data
input_data = sample_data.drop('HP', axis=1).to_dict()
# Get the actual HP
actual_hp = sample_data['HP'].values[0]

# Predict the HP using the function
predicted_hp = predict_pokemon_hp(input_data)

# Check if the predicted HP is close to the actual HP
# We use a tolerance of 15.909 as it is the mean absolute error of the model
assert abs(predicted_hp - actual_hp) <= 15.909, 'The predicted HP is not close to the actual HP'