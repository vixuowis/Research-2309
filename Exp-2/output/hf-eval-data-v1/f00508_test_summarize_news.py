def test_summarize_news():
    """
    This function tests the 'summarize_news' function by using a sample news article.
    """
    # Define a long news article
    long_news_article = 'Dal 31 maggio è infine partita la piattaforma ITsART, a più di un anno da quando – durante il primo lockdown – il ministro della Cultura Dario Franceschini ne aveva parlato come di «una sorta di Netflix della cultura», pensata per «offrire a tutto il mondo la cultura italiana a pagamento». È presto per dare giudizi definitivi sulla piattaforma, e di certo sarà difficile farlo anche più avanti senza numeri precisi. Al momento, l’unica cosa che si può fare è guardare com’è fatto il sito, contare quanti contenuti ci sono (circa 700 “titoli”, tra film, documentari, spettacoli teatrali e musicali e altri eventi) e provare a dare un giudizio sul loro valore e sulla loro varietà. Intanto, una cosa notata da più parti è che diversi contenuti di ITsART sono a pagamento sulla piattaforma sebbene altrove, per esempio su RaiPlay, siano invece disponibili gratuitamente.'
    # Call the 'summarize_news' function
    summary = summarize_news(long_news_article)
    # Assert that the summary is not empty
    assert summary != '', 'The summary is empty.'

test_summarize_news()