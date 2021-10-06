from flask import Flask, request
from flask_cors import CORS, cross_origin
import json

from app.models.question import Question
from app.ml_models.question_generation.question_generator import QuestionGenerator

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def hello():
    return json.dumps('Hello to Leaf!')


@app.route("/generate", methods=["POST"])
@cross_origin()
def generate():
    #postman
    # text = request.form['text']

    requestJson = json.loads(request.data)
    text = requestJson['text']
    title = requestJson['title']    

    if requestJson['count'] == '':
        count = 20
    else:
        count = int(requestJson['count'])

    question_generator = QuestionGenerator()
    input_answer = '[MASK]'
    generated = question_generator.generate(input_answer, text)

    answer, question = generated.split('<sep>')

    questions = [
        Question(answer, question, [])
        ]

    result = list(map(lambda x: json.dumps(x.__dict__), questions))

    return json.dumps(result)
    # return json.dumps(questions)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9002, app)