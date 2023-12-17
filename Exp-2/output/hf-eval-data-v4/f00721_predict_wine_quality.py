# requirements_file --------------------

!pip install -U huggingface_hub joblib pandas

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def predict_wine_quality(data_filename='winequality-red.csv', model_filename='sklearn_model.joblib', repo_id='julien-c/wine-quality'):
    """
    Predicts the wine quality as good or bad based on the chemical properties.

    Parameters:
        data_filename (str): The filename of the wine quality dataset.
        model_filename (str): The filename of the pre-trained model.
        repo_id (str): The Hugging Face repository ID.

    Returns:
        tuple: A tuple containing the labels and model accuracy.
    """
    # Load the pre-trained model
    model = joblib.load(cached_download(hf_hub_url(repo_id, model_filename)))

    # Load the dataset
    data_file = cached_download(hf_hub_url(repo_id, data_filename))
    wine_df = pd.read_csv(data_file, sep=';')

    # Separate features and target variable
    X = wine_df.drop(['quality'], axis=1)
    Y = wine_df['quality']

    # Predict and evaluate
    labels = model.predict(X)
    model_score = model.score(X, Y)
    return labels, model_score

# test_function_code --------------------

def test_predict_wine_quality():
    print("Testing predict_wine_quality function...")
    labels, accuracy = predict_wine_quality()

    # Test case: Check if labels are not empty
    assert len(labels) > 0, "No labels predicted."
    # Test case: Check if accuracy is a valid percentage
    assert 0 <= accuracy <= 1, "Accuracy out of valid range."
    print("Testing completed successfully with accuracy: {:.2f}%".format(accuracy * 100))

# Run the test
if __name__ == '__main__':
    test_predict_wine_quality()