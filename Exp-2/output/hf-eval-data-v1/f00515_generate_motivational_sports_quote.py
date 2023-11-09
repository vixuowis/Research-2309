from transformers import pipeline

def generate_motivational_sports_quote():
    '''
    This function generates a motivational sports quote using the Hugging Face Transformers library.
    The function uses the 'TehVenom/PPO_Pygway-V8p4_Dev-6b' model, which is a large-scale generative model, capable of generating high-quality text.
    '''
    # Create a text generation model
    text_generator = pipeline('text-generation', model='TehVenom/PPO_Pygway-V8p4_Dev-6b')
    # Generate a motivational sports quote
    generated_text = text_generator('Motivational quote about sports:', max_length=50)[0]['generated_text']
    return generated_text