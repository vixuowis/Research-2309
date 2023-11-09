from transformers import AutoModelWithLMHead, AutoTokenizer

# Load the tokenizer associated with the pre-trained model
tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')

# Load the pre-trained model
model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')

# Define a function 'gen_sentence' that takes a list of words and a maximum sentence length as input
# This function will tokenize the input words, generate a sentence using the pre-trained model, and then decode the generated sentence back into text using the tokenizer

def gen_sentence(words, max_length=32):
    """
    This function generates a creative and coherent sentence for marketing content using a pre-trained model.
    
    Parameters:
    words (str): A list of words to be included in the sentence.
    max_length (int): The maximum length of the sentence to be generated.
    
    Returns:
    str: A generated sentence.
    """
    # Tokenize the input words
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')
    
    # Generate a sentence using the pre-trained model
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    
    # Decode the generated sentence back into text using the tokenizer
    return tokenizer.decode(output[0], skip_special_tokens=True)