from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM


def translate_property_description(property_description):
    """
    This function translates a property description from English to French using the pre-trained model 'optimum/t5-small'.
    
    Parameters:
    property_description (str): The property description in English.
    
    Returns:
    str: The translated property description in French.
    """
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    
    # Create a translation pipeline
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    
    # Translate the property description
    results = translator(property_description)
    
    return results[0]['translation_text']