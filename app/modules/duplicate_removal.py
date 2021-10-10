from platform import dist
from typing import List
import string
import re

def remove_duplicates(items: List[str]) -> List[str]:
    unique_items = []
    normalized_unique_items = []

    for item in items:
        normalized_item = _normalize_item(item)

        if normalized_item not in normalized_unique_items:
            unique_items.append(item)
            normalized_unique_items.append(normalized_item)

    return unique_items

def remove_distractors_duplicate_with_correct_answer(correct: str, distractors: List[str]):
    for distractor in distractors:
        if _normalize_item(correct) == _normalize_item(distractor):
            distractors.remove(distractor)
    
    return distractors

'''
Code from: https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
'''
def _normalize_item(item) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(item))))


