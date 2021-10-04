from flask import Flask, request
from flask_cors import CORS, cross_origin
import json

from app.models.question import Question

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

    #TODO use the real stuff here
    questions = [Question("Koala", "Which is the cutest animal?", ['Panda', 'Gorrila', 'Dolphin'])]

    result = list(map(lambda x: json.dumps(x.__dict__), questions))

    return json.dumps(result)
    # return json.dumps(questions)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9002, app)