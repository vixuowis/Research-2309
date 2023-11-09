from transformers import pipeline


def assess_paraphrase_adequacy(generated_paraphrase):
    """
    This function uses the 'prithivida/parrot_adequacy_model' from Hugging Face Transformers to assess the adequacy of a paraphrased text.
    
    Parameters:
    generated_paraphrase (str): The paraphrased text to be assessed.
    
    Returns:
    paraphrase_adequacy (dict): The adequacy score of the paraphrased text.
    """
    # Create a text classification model using the pipeline function
    adequacy_classifier = pipeline('text-classification', model='prithivida/parrot_adequacy_model')
    
    # Use the created classifier to assess the adequacy of the paraphrased text
    paraphrase_adequacy = adequacy_classifier(generated_paraphrase)
    
    return paraphrase_adequacy