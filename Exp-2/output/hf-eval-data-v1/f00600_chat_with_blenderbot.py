from transformers import pipeline

# Function to chat with Blenderbot
# Blenderbot is a model trained on a variety of dialogue datasets and is capable of engaging in open-domain conversations
# It can handle discussions on various subjects, displaying knowledge, empathy, and personality as needed
# The function takes a string as input and returns a string as output

def chat_with_blenderbot(text):
    # Create a conversational model using the pipeline function
    # Specify the model 'hyunwoongko/blenderbot-9B' to be loaded
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    # Send the input text to the model and get the response
    response = chatbot(text)
    # Return the response
    return response