from typing import List
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.models.question import Question

#TODO: Call the neeeded ml models to generate the desired count of multiple choice quesitons. Currently using 3 different ML models, but later it could be only 1. 

def generate(context: str, desired_count: int) -> List[Question]:
    #TODO Clean the text

    questions = _generate_answers(context, desired_count)

    for question in questions:
        print(question.answerText)
    
    #TODO Generate questions for those answers
    # question_generator = QuestionGenerator()
    # generated = question_generator.generate(answer, context)
    # answer, question = generated.split('<sep>')

    #TODO Generate distractors for those questions and remove duplicates

    return []

def _generate_answers(context: str, desired_count: int) -> List[Question]:
    answer_generator = AnswerGenerator()

    answers = answer_generator.generate(context, 4)

    #TODO: remove duplicate answers 

    questions = []
    for answer in answers:
        questions.append(Question(answer))

    return questions