def test_detect_conflicting_info():
    sentence_pairs = [('A man is eating pizza', 'A man eats something'), ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')]
    scores = detect_conflicting_info(sentence_pairs)
    assert len(scores) == len(sentence_pairs), 'The number of scores should be equal to the number of sentence pairs.'
    for score in scores:
        assert len(score) == 3, 'Each score should have three elements corresponding to contradiction, entailment, and neutral.'
        assert sum(score) == 1, 'The sum of the scores for each sentence pair should be 1.'

test_detect_conflicting_info()