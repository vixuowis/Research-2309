from transformers import pipeline

def emotion_classifier(text):
    '''
    This function uses the Hugging Face Transformers library to classify the emotion in a given piece of text.
    The model used is 'michellejieli/emotion_text_classifier', which has been fine-tuned on transcripts from the Friends show.
    It predicts 6 Ekman emotions and a neutral class. These emotions include anger, disgust, fear, joy, neutrality, sadness, and surprise.
    
    Args:
    text (str): The text to be classified.
    
    Returns:
    dict: The predicted emotion and its score.
    '''
    classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    result = classifier(text)
    return result