import re

def clean_text(text: str) -> str:
    """Clean the text from symbols and additional information.
    
    Args:
        text (str): The text.
    
    Returns:
        str: CLeaned text.
    """
    cleaned_text = _remove_brackets(text)
    cleaned_text = _remove_square_brackets(cleaned_text)
    cleaned_text = _remove_multiple_spaces(cleaned_text)
    cleaned_text = _replace_weird_hyphen(cleaned_text)
    
    return cleaned_text
    

def _remove_brackets(text: str) -> str:
    """ Remove brackets '(', ')' and the information between them. 

    e.g. "The koala has a body length of 60–85 cm (24–33 in)."
    
    Args:
        text (str): The text.
    
    Returns:
        str: CLeaned text.
    """
    return re.sub(r'\((.*?)\)', lambda L: '', text)


def _remove_square_brackets(text: str) -> str:
    """ Remove square brackets '[', ']' and the information between them. 

    e.g. The koala[1] is cool."
    
    Args:
        text (str): The text.
    
    Returns:
        str: CLeaned text.
    """

    return re.sub(r'\[(.*?)\]', lambda L: '', text)


def _remove_multiple_spaces(text: str) -> str:
    """Remove multiple white spaces. 

    e.g. "The koala         is     angry  !"
    
    Args:
        text (str): The text.
    
    Returns:
        str: CLeaned text.
    """

    return re.sub(' +', ' ', text)


def _replace_weird_hyphen(text: str) -> str:
    """ Replace weird '–' hyphen that's not recognized as a delimeter by spacy. 

    e.g. '4–15 kg' -> '4-15 kg' 
    (You may not see a difference, but there fucking is. This motherfucker '–' is not recognized by spacy as a delimeter.)
    
    Args:
        text (str): The text.
    
    Returns:
        str: CLeaned text.
    """
    return text.replace('–', '-')