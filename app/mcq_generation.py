from typing import List

from app.modules.duplicate_removal import remove_duplicates
from app.modules.text_cleaning import clean_text
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.models.question import Question

import time


class MCQGenerator():
    def __init__(self, is_verbose=False):
        start_time = time.perf_counter()
        print('Loading ML Models...')

        self.answer_generator = AnswerGenerator()
        print('Loaded AnswerGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.question_generator = QuestionGenerator()
        print('Loaded QuestionGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.distractor_generator = DistractorGenerator()
        print('Loaded DistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

    def generate_mcq_questions(self, context: str, desired_count: int) -> List[Question]:
        cleaned_text =  clean_text(context)

        questions = self._generate_answers(cleaned_text, desired_count)
        questions = self._generate_questions(cleaned_text, questions)
        questions = self._generate_distractors(cleaned_text, questions)
        
        for question in questions:
            print('-------------------')
            print(question.answerText)
            print(question.questionText)
            print(question.distractors)

        return questions

    def _generate_answers(self, context: str, desired_count: int) -> List[Question]:
        answers = self.answer_generator.generate(context, desired_count)
        unique_answers = remove_duplicates(answers)

        questions = []
        for answer in unique_answers:
            questions.append(Question(answer))

        return questions

    def _generate_questions(self, context: str, questions: List[Question]) -> List[Question]:
        for question in questions:
            question.questionText = self.question_generator.generate(question.answerText, context)

        return questions

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        for question in questions:
            distractors =  self.distractor_generator.generate(5, question.answerText, question.questionText, context)
            unique_distractors = remove_duplicates(distractors)

            question.distractors = unique_distractors

        return questions