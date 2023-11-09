from transformers import pipeline


def summarize_text(article: str, max_length: int = 130, min_length: int = 30, do_sample: bool = False) -> str:
    """
    Summarize a given text using the Hugging Face Transformers library.

    Args:
        article (str): The text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 130.
        min_length (int, optional): The minimum length of the summary. Defaults to 30.
        do_sample (bool, optional): Whether or not to use sampling in the generation process. Defaults to False.

    Returns:
        str: The summarized text.
    """
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['summary_text']
    return summary