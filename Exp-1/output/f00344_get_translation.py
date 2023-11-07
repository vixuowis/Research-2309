from typing import *
from typing import Dict, List, Optional

def get_translation(books: Dict[str, List[Dict[str, str]]], lang: str, index: int) -> Optional[str]:
    """
    Get the translation of a book at a specific index and language.
    
    Args:
        books (Dict[str, List[Dict[str, str]]]): A dictionary containing books categorized by type.
        lang (str): The language of the translation.
        index (int): The index of the book to retrieve the translation from.
    
    Returns:
        Optional[str]: The translation of the book in the specified language. Returns None if the index is out of range or the translation does not exist.
    """
    if 'train' in books and index < len(books['train']):
        translation = books['train'][index].get('translation')
        if translation and lang in translation:
            return translation[lang]
    return None
