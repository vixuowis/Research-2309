from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to translate French to English using Hugging Face Transformers

def translate_french_to_english(sentence):
    '''
    Translates a French sentence to English using the 'bigscience/bloomz-560m' model from Hugging Face Transformers.

    Parameters:
    sentence (str): The French sentence to translate.

    Returns:
    str: The translated English sentence.
    '''
    # Define the model checkpoint
    checkpoint = 'bigscience/bloomz-560m'

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    # Encode the sentence with translation instruction
    inputs = tokenizer.encode(f"Translate to English: {sentence}", return_tensors='pt')

    # Generate the output
    outputs = model.generate(inputs)

    # Decode the output to get the translated sentence
    translated_sentence = tokenizer.decode(outputs[0])

    return translated_sentence