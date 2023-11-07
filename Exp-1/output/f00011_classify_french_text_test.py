from f00011_classify_french_text import *
def test_classify_french_text():
    text = "Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."
    expected_result = [{'label': '5 stars', 'score': 0.7273}]
    assert classify_french_text(text) == expected_result

test_classify_french_text()
