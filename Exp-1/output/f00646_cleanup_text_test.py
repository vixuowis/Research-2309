from f00646_cleanup_text import *
>>> inputs = {"normalized_text": "àçèëíïöü"}
>>> expected_output = {"normalized_text": "aceeiiou"}
>>> assert cleanup_text(inputs) == expected_output

>>> inputs = {"normalized_text": "àçèëíïöüàçèëíïöü"}
>>> expected_output = {"normalized_text": "aceeiiouaceeiiou"}
>>> assert cleanup_text(inputs) == expected_output

>>> inputs = {"normalized_text": "àçèëíïöüàçèëíïöüàçèëíïöü"}
>>> expected_output = {"normalized_text": "aceeiiouaceeiiouaceeiiou"}
>>> assert cleanup_text(inputs) == expected_output
