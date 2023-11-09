# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_text(article_text):
    """
    Summarize a given long article text.

    Args:
        article_text (str): The long article text to be summarized.

    Returns:
        str: The summarized text.
    """
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')
    inputs = tokenizer.encode(article_text, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_text():
    """
    Test the summarize_text function.
    """
    article_text = 'India wicket-keeper batsman Rishabh Pant has said someone from the crowd threw a ball on pacer Mohammed Siraj while he was fielding in the ongoing third Test against England on Wednesday. Pant revealed the incident made India skipper Virat Kohli upset. I think, somebody threw a ball inside, at Siraj, so he [Kohli] was upset, said Pant in a virtual press conference after the close of the first day's play.You can say whatever you want to chant, but don't throw things at the fielders and all those things. It is not good for cricket, I guess, he added.In the third session of the opening day of the third Test, a section of spectators seemed to have asked Siraj the score of the match to tease the pacer. The India pacer however came with a brilliant reply as he gestured 1-0 (India leading the Test series) towards the crowd.Earlier this month, during the second Test match, there was some bad crowd behaviour on a show as some unruly fans threw champagne corks at India batsman KL Rahul.Kohli also intervened and he was seen gesturing towards the opening batsman to know more about the incident. An over later, the TV visuals showed that many champagne corks were thrown inside the playing field, and the Indian players were visibly left frustrated.Coming back to the game, after bundling out India for 78, openers Rory Burns and Haseeb Hameed ensured that England took the honours on the opening day of the ongoing third Test.At stumps, England's score reads 120/0 and the hosts have extended their lead to 42 runs. For the Three Lions, Burns (52) and Hameed (60) are currently unbeaten at the crease.Talking about the pitch on opening day, Pant said, They took the heavy roller, the wicket was much more settled down, and they batted nicely also, he said. But when we batted, the wicket was slightly soft, and they bowled in good areas, but we could have applied [ourselves] much better.Both England batsmen managed to see off the final session and the hosts concluded the opening day with all ten wickets intact, extending the lead to 42.(ANI)'
    summary = summarize_text(article_text)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) < len(article_text), 'The summary should be shorter than the original text.'

# call_test_function_code --------------------

test_summarize_text()