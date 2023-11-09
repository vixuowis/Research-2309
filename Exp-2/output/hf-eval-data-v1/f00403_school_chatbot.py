from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# Function to generate response from the chatbot
# Input: User query as a string
# Output: Chatbot response as a string
def school_chatbot(input_text):
    # Load the pre-trained model and tokenizer
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot_small-90M')

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response