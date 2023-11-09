from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

def get_animal_info(query):
    '''
    This function uses the Hugging Face Transformers library to answer questions about animals based on a predefined table.
    The table contains information about different animals and their characteristics.
    The function uses the 'neulab/omnitab-large-finetuned-wtq' model from Hugging Face Transformers.
    '''
    # Instantiate the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-finetuned-wtq')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-finetuned-wtq')

    # Define the table data
    data = {
        'Animal': ['Tiger', 'Lion', 'Giraffe', 'Elephant'],
        'Habitat': ['Forest', 'Grassland', 'Savanna', 'Savanna'],
        'Average Lifespan': [10, 12, 25, 50],
    }
    table = pd.DataFrame.from_dict(data)

    # Use the tokenizer to create an encoding of the table and the query
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Use the model to generate an output based on the encoding
    outputs = model.generate(**encoding)

    # Decode the output to get the final answer to the question
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)