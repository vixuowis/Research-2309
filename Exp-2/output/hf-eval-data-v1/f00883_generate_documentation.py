from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline


def generate_documentation(tokenized_code):
    """
    Generate documentation for a given Python function using a pretrained model.

    Args:
        tokenized_code (str): The Python function to document, as a string.

    Returns:
        str: The generated documentation for the function.
    """
    pipeline = SummarizationPipeline(
        model=AutoModelWithLMHead.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_python'),
        tokenizer=AutoTokenizer.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_python', skip_special_tokens=True),
        device=0
    )
    return pipeline([tokenized_code])[0]['summary_text']