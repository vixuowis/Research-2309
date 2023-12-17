# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def summarize_french_article(article_text: str) -> str:
    """
    Summarizes a given French article using a pre-trained T5 model.
    
    Parameters:
    article_text (str): The French article text to summarize.
    
    Returns:
    str: The summarized article text in French.
    """
    tokenizer = T5Tokenizer.from_pretrained('plguillou/t5-base-fr-sum-cnndm')
    model = T5ForConditionalGeneration.from_pretrained('plguillou/t5-base-fr-sum-cnndm')
    input_text = 'summarize: ' + article_text
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(input_tokens)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_french_article():
    print("Testing started.")
    article_text = "Selon un rapport récent, les constructeurs automobiles prévoient d'accélérer la production de ...
"
    expected_summary_part = " ... " # Expected summary content part for validation

    print("Testing summarization.")
    summary = summarize_french_article(article_text)
    assert expected_summary_part in summary, f"Summary does not contain expected part: {expected_summary_part}"
    print("Testing finished.")
    return True