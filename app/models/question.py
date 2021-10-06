from typing import List

class Question:
    def __init__(self, answerText:str, questionText: str = '', distractors: List[str] = []):
        self.answerText = answerText
        self.questionText = questionText
        self.distractors = distractors
