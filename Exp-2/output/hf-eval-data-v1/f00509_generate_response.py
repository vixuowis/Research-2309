from transformers import AutoModelForCausalLM

# Function to generate response using the pre-trained conversational model
# Input: User's question as a string
# Output: Model's response as a string
def generate_response(input_query):
    # Load the pre-trained conversational model
    conversation_bot = AutoModelForCausalLM.from_pretrained('Zixtrauce/JohnBot')
    # Generate a response to the user's question
    output_query = conversation_bot.generate_response(input_query)
    return output_query