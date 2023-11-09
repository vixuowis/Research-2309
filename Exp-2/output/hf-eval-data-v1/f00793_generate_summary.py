from transformers import pipeline


def generate_summary(article_text: str, max_length: int = 50, num_return_sequences: int = 1) -> str:
    """
    Generate a brief summary for a given article using GPT-2 Large model.

    Args:
        article_text (str): The first few sentences of the news article.
        max_length (int, optional): The maximum length of the generated summary. Defaults to 50.
        num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.

    Returns:
        str: The generated summary.
    """
    summary_generator = pipeline('text-generation', model='gpt2-large')
    summary = summary_generator(article_text, max_length=max_length, num_return_sequences=num_return_sequences)[0]['generated_text']
    return summary