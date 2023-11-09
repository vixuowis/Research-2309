from transformers import pipeline


def complete_sentence(sentence):
    '''
    This function uses the Hugging Face Transformers library to complete a sentence using the 'roberta-base' model.
    The model is a transformers model pretrained on a large corpus of English data in a self-supervised fashion using the Masked language modeling (MLM) objective.
    It is case-sensitive and can be fine-tuned on a downstream task.
    
    Parameters:
    sentence (str): The sentence to be completed. The word to be filled should be replaced with '<mask>'.
    
    Returns:
    str: The completed sentence.
    '''
    unmasker = pipeline('fill-mask', model='roberta-base')
    completed_sentence = unmasker(sentence)[0]['sequence']
    return completed_sentence