from f00224_convert_ner_tags import *
def test_convert_ner_tags():
    assert convert_ner_tags([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == ["O", "B-corporation", "I-corporation", "B-creative-work", "I-creative-work", "B-group", "I-group", "B-location", "I-location", "B-person"]
    assert convert_ner_tags([12, 11, 10]) == ["I-product", "B-product", "I-person"]
    assert convert_ner_tags([]) == []

if __name__ == "__main__":
    test_convert_ner_tags()
