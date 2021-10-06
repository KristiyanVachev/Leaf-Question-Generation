from typing import List
from app.models.question import Question

#TODO: Call the neeeded ml models to generate the desired count of multiple choice quesitons. Currently using 3 different ML models, but later it could be only 1. 

def generate(context: str, desired_count: int) -> List[Question]:
    #TODO Clean the text
    #TODO generate answers and remove duplicates
    #TODO Generate questions for those answers
    #TODO Generate distractors for those questions and remove duplicates
    pass