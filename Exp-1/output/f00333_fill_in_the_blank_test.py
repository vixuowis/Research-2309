from f00333_fill_in_the_blank import *
def test_fill_in_the_blank():
    text = "The Milky Way is a <mask> galaxy."
    expected_result = "The Milky Way is a spiral galaxy."
    assert fill_in_the_blank(text) == expected_result

test_fill_in_the_blank()
