def test_translate_swedish_to_english():
    input_text = 'Stockholm är Sveriges huvudstad och största stad. Den har en rik historia och erbjuder många kulturella och historiska sevärdheter.'
    expected_output = 'Stockholm is the capital and largest city of Sweden. It has a rich history and offers many cultural and historical attractions.'
    assert translate_swedish_to_english(input_text) == expected_output

test_translate_swedish_to_english()