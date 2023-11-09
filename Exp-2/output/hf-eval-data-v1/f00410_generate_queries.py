from transformers import T5Tokenizer, T5ForConditionalGeneration


def generate_queries(document):
    """
    This function takes a document as input and generates possible user queries based on the document.
    It uses the T5 model trained on the MS MARCO dataset for generating queries from documents.
    
    Args:
    document (str): The document for which to generate queries.
    
    Returns:
    str: The generated queries.
    """
    # Load the pre-trained model and its corresponding tokenizer
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    
    # Tokenize the input document
    input_ids = tokenizer.encode(document, return_tensors='pt')
    
    # Perform text-to-text generation using the T5 language model
    generated_ids = model.generate(input_ids)
    
    # Decode the generated tokens back into text format
    generated_queries = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_queries