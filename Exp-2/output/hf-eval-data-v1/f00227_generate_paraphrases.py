from parrot import Parrot
import torch


def generate_paraphrases(phrase):
    '''
    This function generates paraphrases of a given phrase using the Parrot paraphraser.
    
    Parameters:
    phrase (str): The phrase to be paraphrased.
    
    Returns:
    list: A list of paraphrased phrases.
    '''
    # Create a Parrot instance with the model_tag set to 'prithivida/parrot_paraphraser_on_T5' and use_gpu set to False if you don't have a GPU available.
    parrot = Parrot(model_tag='prithivida/parrot_paraphraser_on_T5', use_gpu=False)
    # Use the augment method of the Parrot instance to generate paraphrased sentences from the given input phrase.
    para_phrases = parrot.augment(input_phrase=phrase)
    # Return the list of paraphrased phrases
    return para_phrases