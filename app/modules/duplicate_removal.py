from platform import dist
from typing import List
import string
import re
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def remove_duplicates(items: List[str]) -> List[str]:
    unique_items = []
    normalized_unique_items = []

    for item in items:
        normalized_item = _normalize_item(item)

        if normalized_item not in normalized_unique_items:
            unique_items.append(item)
            normalized_unique_items.append(normalized_item)

    return unique_items

def remove_distractors_duplicate_with_correct_answer(correct: str, distractors: List[str]) -> List[str]:
    for distractor in distractors:
        if _normalize_item(correct) == _normalize_item(distractor):
            distractors.remove(distractor)
    
    return distractors

def _get_most_distinct_from_key(key: str, items: List[str]) -> List[str]:
    #TODO: This seems not as useful. For example "the family Phascolarctidae" and "the family Vombatidae" are close, but good distractors. 

    return items

def _get_most_distinct_from_each_other():
    #TODO 
    # calculate bleu for each with each.
    # find the most similar pair
    # remove the second in the original list (assuming the list comes ordered by better)
    # run until you get the desired amount 
    pass

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

def _calculate_nltk_bleu(references: List[str], hypothesis: str, bleu_n: int = 1):
    if hypothesis == '': 
        return 0, 0, 0, 0 

    # Word tokenize
    refs_tokenized = list(map(lambda x: word_tokenize(x), references))
    hyp_tokenized = word_tokenize(hypothesis)

    # Smoothing function to avoid the cases where it resuts 1.0 in the cases when // Corpus/Sentence contains 0 counts of 2-gram overlaps. BLEU scores might be undesirable; use SmoothingFunction() //
    chencherry = SmoothingFunction()
    bleu = 0

    if bleu_n == 1:
        bleu = sentence_bleu(refs_tokenized, hyp_tokenized, weights=(1, 0, 0, 0), smoothing_function=chencherry.method2)
    elif bleu_n == 2:
        bleu = sentence_bleu(refs_tokenized, hyp_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method2)
    elif bleu_n == 3: 
        bleu = sentence_bleu(refs_tokenized, hyp_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method2)
    elif bleu_n == 4:
        bleu = sentence_bleu(refs_tokenized, hyp_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method2)

    return bleu