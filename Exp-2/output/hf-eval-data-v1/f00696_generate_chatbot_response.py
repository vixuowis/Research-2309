from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Function to generate chatbot response
# This function uses the Hugging Face Transformers library to load a pre-trained model and tokenizer
# The model is used to generate a response to a user's input message
# The tokenizer is used to encode the input message and decode the generated response

def generate_chatbot_response(input_message):
    # Load the pre-trained model
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-1B-distill')
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-1B-distill')
    # Encode the input message
    inputs = tokenizer(input_message, return_tensors='pt')
    # Generate a response
    outputs = model.generate(inputs['input_ids'])
    # Decode the generated response
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return the decoded output
    return decoded_output