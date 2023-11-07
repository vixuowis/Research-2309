from f00363_translate_text import *
def test_translate_text():
    text = "Legumes share resources with nitrogen-fixing bacteria."
    expected_output = "Les légumes partagent des ressources avec des bactéries fixatrices d'azote."

    output = translate_text(text)

    assert output == expected_output, f"Expected: {expected_output}, but got: {output}"


test_translate_text()
