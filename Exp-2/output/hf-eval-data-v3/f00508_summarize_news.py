# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_news(article_text):
    """
    Summarize a long news article using the 'it5/it5-base-news-summarization' model from Hugging Face Transformers.

    Args:
        article_text (str): The text of the news article to be summarized.

    Returns:
        str: The summarized text of the news article.
    """
    summarizer = pipeline('summarization', model='it5/it5-base-news-summarization')
    summary = summarizer(article_text)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_news():
    """
    Test the summarize_news function with some example articles.
    """
    article1 = 'Dal 31 maggio è infine partita la piattaforma ITsART, a più di un anno da quando – durante il primo lockdown – il ministro della Cultura Dario Franceschini ne aveva parlato come di «una sorta di Netflix della cultura», pensata per «offrire a tutto il mondo la cultura italiana a pagamento». È presto per dare giudizi definitivi sulla piattaforma, e di certo sarà difficile farlo anche più avanti senza numeri precisi. Al momento, l’unica cosa che si può fare è guardare com’è fatto il sito, contare quanti contenuti ci sono (circa 700 “titoli”, tra film, documentari, spettacoli teatrali e musicali e altri eventi) e provare a dare un giudizio sul loro valore e sulla loro varietà. Intanto, una cosa notata da più parti è che diversi contenuti di ITsART sono a pagamento sulla piattaforma sebbene altrove, per esempio su RaiPlay, siano invece disponibili gratuitamente.'
    assert len(summarize_news(article1)) < len(article1)
    article2 = 'This is a short article. It should not be summarized.'
    assert summarize_news(article2) == article2
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_news()