from transformers import AutoModelWithLMHead, AutoTokenizer

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')

def gen_sentence(words, max_length=32):
    """
    This function generates a creative sentence using the provided words.
    It uses a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    words (str): The words to be included in the sentence.
    max_length (int): The maximum length of the generated sentence.
    
    Returns:
    str: The generated sentence.
    """
    # Prepare the input text
    input_text = words
    # Tokenize the input text
    features = tokenizer([input_text], return_tensors='pt')
    # Generate the sentence
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    # Decode the generated sentence and return it
    return tokenizer.decode(output[0], skip_special_tokens=True)