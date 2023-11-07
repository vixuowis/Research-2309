from f00817_classify_sentence_pairs import *
def test_classify_sentence_pairs():
    sentences = [
        "This is the first sentence.",
        "This is the second sentence.",
        "These sentences are paraphrases.",
        "These sentences are not paraphrases."
    ]
    expected_results = [
        {"label": "not paraphrase", "score": 0.987654},
        {"label": "not paraphrase", "score": 0.987321},
        {"label": "paraphrase", "score": 0.987123},
        {"label": "not paraphrase", "score": 0.987987}
    ]

    results = classify_sentence_pairs(sentences)

    assert results == expected_results


test_classify_sentence_pairs()
