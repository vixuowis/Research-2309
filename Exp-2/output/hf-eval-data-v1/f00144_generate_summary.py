from transformers import pipeline

# Function to generate a summary of a given text using the LLaMA-13B model
# from Hugging Face Transformers.
# The function takes a string as input and returns a string as output.
def generate_summary(text):
    # Initialize the text-generation pipeline with the LLaMA-13B model
    generator = pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    # Generate the summary
    summary = generator(text)
    # Return the summary
    return summary