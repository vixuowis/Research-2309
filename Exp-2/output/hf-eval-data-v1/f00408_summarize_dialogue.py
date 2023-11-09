from transformers import LEDForConditionalGeneration, LEDTokenizer


def summarize_dialogue(input_text):
    """
    This function takes a dialogue as input and returns a summarized version of it.
    It uses the pre-trained model 'MingZhong/DialogLED-base-16384' from Hugging Face Transformers, which is built on the Longformer-Encoder-Decoder (LED) architecture and specializes in long dialogue understanding and summarization.
    
    Parameters:
    input_text (str): The dialogue to be summarized.
    
    Returns:
    str: The summarized dialogue.
    """
    # Load the pre-trained model
    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384')
    # Initialize an LEDTokenizer with the same model
    tokenizer = LEDTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')
    # Prepare the input text by tokenizing it
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    # Generate the summary from the input tokens
    summary_ids = model.generate(input_tokens)
    # Decode the generated summary tokens
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary