from f00364_pipeline import *
translator = pipeline("translation", model="my_awesome_opus_books_model")
translator(text)
