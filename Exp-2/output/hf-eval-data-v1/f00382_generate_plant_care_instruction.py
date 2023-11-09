from transformers import pipeline

# This function generates a short and simple plant care instruction using a pre-trained GPT model.
def generate_plant_care_instruction(prompt):
    # Initialize the pipeline with 'text-generation' as task and the pre-trained GPT model.
    generator = pipeline('text-generation', model='gpt2')
    # Generate the text by providing the input prompt.
    instructions = generator(prompt)
    return instructions[0]['generated_text']