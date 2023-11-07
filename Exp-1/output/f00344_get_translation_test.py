from f00344_get_translation import *
def test_get_translation():
    books = {
        'train': [
            {
                'id': '90560',
                'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
                                 'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}
            }
        ]
    }
    assert get_translation(books, 'en', 0) == 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.'
    assert get_translation(books, 'fr', 0) == 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'
    assert get_translation(books, 'es', 0) == None
    assert get_translation(books, 'en', 1) == None
    assert get_translation(books, 'fr', -1) == None

test_get_translation()
