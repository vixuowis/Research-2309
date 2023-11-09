import joblib
import pandas as pd

# Function to classify movie reviews
# Input: Path to the CSV file containing movie reviews
# Output: Predictions for each review (Positive or Negative)
def classify_movie_reviews(data_path):
    # Load the pretrained binary classification model
    model = joblib.load('model.joblib')
    
    # Load the movie review dataset
    data = pd.read_csv(data_path)
    
    # Use the model to predict the sentiment of the reviews
    predictions = model.predict(data)
    
    return predictions