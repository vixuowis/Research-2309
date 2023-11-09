from transformers import pipeline

# Function to fill in the gaps in a sentence using the DebertaModel from Hugging Face Transformers
# The function takes a sentence with a '[MASK]' placeholder and returns the sentence with the placeholder filled
# The model used is 'microsoft/deberta-v3-base'
def fill_mask(sentence):
    fill_mask = pipeline('fill-mask', model='microsoft/deberta-v3-base')
    result = fill_mask(sentence)
    return result