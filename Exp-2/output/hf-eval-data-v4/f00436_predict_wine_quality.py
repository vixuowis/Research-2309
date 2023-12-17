# requirements_file --------------------

!pip install -U huggingface_hub joblib pandas

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def predict_wine_quality():
    """
    Load the wine quality classification model and dataset.
    Predict the quality of the wines and return the predicted labels.

    Returns:
        labels (array): Predicted wine quality labels.
    """
    # Load the model from Hugging Face Hub
    REPO_ID = "julien-c/wine-quality"
    FILENAME = "sklearn_model.joblib"
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))

    # Load the dataset
    data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))
    wine_df = pd.read_csv(data_file, sep=";")
    X = wine_df.drop(['quality'], axis=1)
    
    # Predict wine quality
    labels = model.predict(X)
    return labels

# test_function_code --------------------

def test_predict_wine_quality():
    print("Testing started.")
    labels = predict_wine_quality()
    
    # 测试用例 1：检查预测结果是否为非空
    print("Testing case [1/1] started.")
    assert len(labels) > 0, f"Test case [1/1] failed: The prediction should not be empty."
    print("Testing finished.")

# 运行测试函数
test_predict_wine_quality()