from platform import dist
import textwrap
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator

def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    
    print('Context:')

    for wrap in textwrap.wrap(context, width=120):
        print(wrap)
    print()

    print('Question:')
    print(generated)

    print('Answer:')
    print(answer)
    print('-----------------------------')


context = '''The koala or, inaccurately, koala bear[a] (Phascolarctos cinereus), is an arboreal herbivorous marsupial native to Australia. It is the only extant representative of the family Phascolarctidae and its closest living relatives are the wombats, which are members of the family Vombatidae. The koala is found in coastal areas of the mainland's eastern and southern regions, inhabiting Queensland, New South Wales, Victoria, and South Australia. It is easily recognisable by its stout, tailless body and large head with round, fluffy ears and large, spoon-shaped nose. The koala has a body length of 60–85 cm (24–33 in) and weighs 4–15 kg (9–33 lb). Fur colour ranges from silver grey to chocolate brown. Koalas from the northern populations are typically smaller and lighter in colour than their counterparts further south. These populations possibly are separate subspecies, but this is disputed.'''
real_answer = 'Koala'
input_answer = '[MASK]'

#TODO 
question_generator = QuestionGenerator()
generated = question_generator.generate(input_answer, context)
answer, question = generated.split('<sep>')
show_result(generated, real_answer, context)

# Distractors
distractor_generator = DistractorGenerator()
distractors = distractor_generator.generate(1, answer, question, context)

print(distractors)