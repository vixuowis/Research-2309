from transformers import pipeline

# Function to predict the HP of a Pokemon character based on its attributes
# The function uses a pre-trained model from Hugging Face
# The model is a regression model trained on the julien-c/kaggle-rounakbanik-pokemon dataset
# The function takes as input a dictionary with the Pokemon attributes and returns the predicted HP

def predict_pokemon_hp(input_data):
    # Initialize the regression model
    regression_model = pipeline('regression', model='julien-c/pokemon-predict-hp')
    # Predict the HP of the Pokemon
    predicted_hp = regression_model(input_data)[0]['score']
    return predicted_hp