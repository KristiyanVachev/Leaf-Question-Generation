from typing import List
from nltk.tokenize import sent_tokenize
import toolz

from app.modules.duplicate_removal import remove_distractors_duplicate_with_correct_answer, remove_duplicates
from app.modules.text_cleaning import clean_text
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.ml_models.sense2vec_distractor_generation.sense2vec_generation import Sense2VecDistractorGeneration
from app.models.question import Question

import time


class MCQGenerator():
    def __init__(self, is_verbose=False):
        start_time = time.perf_counter()
        print('Loading ML Models...')

        # Currently not used
        # self.answer_generator = AnswerGenerator()
        # print('Loaded AnswerGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.question_generator = QuestionGenerator()
        print('Loaded QuestionGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.distractor_generator = DistractorGenerator()
        print('Loaded DistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration()
        print('Loaded Sense2VecDistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

    # Main function
    def generate_mcq_questions(self, context: str, desired_count: int) -> List[Question]:
        cleaned_text =  clean_text(context)

        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)
        
        for question in questions:
            print('-------------------')
            print(question.answerText)
            print(question.questionText)
            print(question.distractors)

        return questions

    def _generate_answers(self, context: str, desired_count: int) -> List[Question]:
        # answers = self.answer_generator.generate(context, desired_count)
        answers = self._generate_multiple_answers_according_to_desired_count(context, desired_count)

        print(answers)
        unique_answers = remove_duplicates(answers)

        questions = []
        for answer in unique_answers:
            questions.append(Question(answer))

        return questions

    def _generate_questions(self, context: str, questions: List[Question]) -> List[Question]:        
        for question in questions:
            question.questionText = self.question_generator.generate(question.answerText, context)

        return questions

    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        context_splits = self._split_context_according_to_desired_count(context, desired_count)

        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate_qna(split)
            questions.append(Question(answer.capitalize(), question))

        questions = list(toolz.unique(questions, key=lambda x: x.answerText))

        return questions

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        for question in questions:
            t5_distractors =  self.distractor_generator.generate(5, question.answerText, question.questionText, context)

            if len(t5_distractors) < 3:
                s2v_distractors = self.sense2vec_distractor_generator.generate(question.answerText, 3)
                distractors = t5_distractors + s2v_distractors
            else:
                distractors = t5_distractors


            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)
            #TODO - filter distractors having a similar bleu score with another distractor

            question.distractors = distractors

        return questions

    # Helper functions 
    def _generate_answer_for_each_sentence(self, context: str) -> List[str]:
        sents = sent_tokenize(context)

        answers = []
        for sent in sents:
            answers.append(self.answer_generator.generate(sent, 1)[0])

        return answers

    #TODO: refactor to create better splits closer to the desired amount
    def _split_context_according_to_desired_count(self, context: str, desired_count: int) -> List[str]:
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count

        context_splits = []

        if sent_ratio < 1:
            return sents
        else:
            take_sents_count = int(sent_ratio + 1)

            start_sent_index = 0

            while start_sent_index < len(sents):
                context_split = ' '.join(sents[start_sent_index: start_sent_index + take_sents_count])
                context_splits.append(context_split)
                start_sent_index += take_sents_count - 1

        return context_splits
    
