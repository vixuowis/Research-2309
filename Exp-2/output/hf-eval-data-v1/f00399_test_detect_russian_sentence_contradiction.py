def test_detect_russian_sentence_contradiction():
    # Test cases with known outcomes
    assert detect_russian_sentence_contradiction('Москва - столица России.', 'Санкт-Петербург - столица России.')  # Expected outcome: True (contradiction)
    assert not detect_russian_sentence_contradiction('Москва - столица России.', 'Москва - крупнейший город России.')  # Expected outcome: False (no contradiction)

test_detect_russian_sentence_contradiction()