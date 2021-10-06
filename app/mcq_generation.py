from typing import List
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.models.question import Question

#TODO: Call the neeeded ml models to generate the desired count of multiple choice quesitons. Currently using 3 different ML models, but later it could be only 1. 

def generate(context: str, desired_count: int) -> List[Question]:
    #TODO Clean the text

    #TODO generate answers and remove duplicates
    answer_generator = AnswerGenerator()
    answer = answer_generator.generate(context, 4)

    #TODO: remove duplicate answers 

    print(answer)
    #TODO Generate questions for those answers

    #TODO Generate distractors for those questions and remove duplicates

    return []