def predict_vacation_success(destination, accommodation, travel_style):
    """
    This function predicts whether a client's vacation will be successful based on their chosen destination, accommodation, and travel style.
    It uses a pre-trained model from Hugging Face's model hub.
    
    Parameters:
    destination (str): The chosen destination of the client.
    accommodation (str): The chosen accommodation of the client.
    travel_style (str): The chosen travel style of the client.
    
    Returns:
    int: A prediction of whether the vacation will be successful (1) or not (0).
    """
    from huggingface_hub import hf_hub_url, cached_download
    import joblib
    import pandas as pd
    from tensorflow.keras.models import load_model

    REPO_ID = 'danupurnomo/dummy-titanic'
    PIPELINE_FILENAME = 'final_pipeline.pkl'
    TF_FILENAME = 'titanic_model.h5'

    model_pipeline = joblib.load(cached_download(hf_hub_url(REPO_ID, PIPELINE_FILENAME)))
    model_seq = load_model(cached_download(hf_hub_url(REPO_ID, TF_FILENAME)))

    new_data = pd.DataFrame({"destination": [destination], "accommodation": [accommodation], "travel_style": [travel_style]})
    prediction = model_seq.predict(model_pipeline.transform(new_data))
    success = (prediction > 0.5).astype(int)
    return success[0]