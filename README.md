
# Leaf: Multiple-Choice Question Generation

Easy to use and understand multiple-choice question generation algorithm using  [T5 Transformers](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html).  The application accepts a short passage of text and uses two fine-tuned T5 Transformer models to first generate multiple **question-answer pairs** corresponding to the given text, after which it uses them to generate ***distractors***  -  additional options used to confuse the test taker.



![question generation process](https://i.ibb.co/fQwPZZv/qg-process.jpg "question generation process")

Originally inspired by a Bachelor's machine learning course ([github link](https://github.com/KristiyanVachev/Question-Generation)) and then continued as a topic for my Master's thesis at Sofia University, Bulgaria. 

## ECIR 2022 Demonstration paper
This work has been accepted as a demo paper for the [ECIR 2022 conference.](https://ecir2022.org/) 

**Video demonstration:** [here](https://www.youtube.com/watch?v=tpxl-UnfmQc)

**Live demo:** *coming soon*

**Paper:** [here](https://arxiv.org/abs/2201.09012)

*Abstract:*
Testing with quiz questions has proven to be an effective strategy for better educational processes. However, manually creating quizzes is a tedious and time-consuming task.  To address this challenge, we present Leaf, a system for generating multiple-choice questions from factual text. In addition to being very well suited for classroom settings, Leaf could be also used in an industrial setup, e.g., to facilitate onboarding and knowledge sharing, or as a component of chatbots, question answering systems, or Massive Open Online Courses (MOOCs).

## Generating question and answer pairs
To generate the question-answer pairs we have fine-tuned a T5 transformer model from [huggingface](https://huggingface.co/transformers/model_doc/t5.html) on the [SQuAD1.1. dataset](https://rajpurkar.github.io/SQuAD-explorer/) which is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles.

The model accepts the target answer and context as input:

    'answer' + '<sep> + 'context' 

and outputs a question that answers the given answer for the corresponding text. 

    'answer' + '<sep> + 'question' 


To allow us to generate question-answer pairs without providing a target answer, we have trained the algorithm to do so when in place of the target answer the '[MASK]' token is passed. 

    '[MASK]' + '<sep> + 'context' 

The full training script can be found in the `training` directory or accessed directly in [Google Colab](https://colab.research.google.com/drive/15GAaD-33jw81sugeBFj_Bp9GkbE_N6E1?usp=sharing). 


## Generating incorrect options  (distractors) 
To generate the distractors, another [T5 transformer model](https://huggingface.co/transformers/model_doc/t5.html)   has been fine-tuned. This time using the [RACE dataset](https://huggingface.co/datasets/race) which consists of more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students.

The model accepts the target answer, question and context as input:

    'answer' + '<sep> + 'question' + 'context' 

and outputs 3 distractors separated by the `'<sep>'` token.

    'distractor1' + '<sep> + 'distractor2' + '<sep> 'distractor3' 


The full training script can be found in the `training` directory or accessed directly in [Google Colab](https://colab.research.google.com/drive/1kWZviQVx1BbelWp0rwZX7H3GIPS7_ZrP?usp=sharing). 

To extend the variety of distractors with simple words that are not so closely related to the context, we have also used [sense2vec](https://pypi.org/project/sense2vec/) word embeddings in the cases where the T5 model does not good enough distractors. 


## Web application
To demonstrate the algorithm, a simple Angular web application has been created. It accepts the given paragraph along with the desired number of questions and outputs each generated question with the ability to redact them (shown below). The algorithm is exposing a simple REST API using *flask* which is consumed by the web app.


![question generation process](https://i.ibb.co/WFJjCgH/1-edited-fullscreen.png "Web application ")

The code for the web application is located in a separated repository [here](https://github.com/KristiyanVachev/QGT-FrontEnd). 





## Installation guide

### Creating a virtual environment *(optional)*
To avoid any conflicts with python packages from other projects, it is a good practice to create a [virtual environment](https://docs.python.org/3/library/venv.html) in which the packages will be installed. If you do not want to this you can skip the next commands and directly install the the requirements.txt file. 

Create a virtual environment :

    python -m venv venv

Enter the virtual environment:

*Windows:*

    . .\venv\Scripts\activate

*Linux or MacOS*

    source .\venv\Scripts\activate

### Installing packages

    pip install -r .\requirements.txt 

### Downloading data

#### Question-answer model
Download the [multitask-qg-ag model](https://drive.google.com/file/d/1-vqF9olcYOT1hk4HgNSYEdRORq-OD5CF/view?usp=sharing) checkpoint and place it in the  `app/ml_models/question_generation/models/` directory.

#### Distractor generation 
Download the [race-distractors model](https://drive.google.com/file/d/1jKdcbc_cPkOnjhDoX4jMjljMkboF-5Jv/view?usp=sharing) checkpoint and place it in the  `app/ml_models/distractor_generation/models/` directory.

Download [sense2vec](https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz), extract it and place the `s2v_old`  folder  and place it in the `app/ml_models/sense2vec_distractor_generation/models/` directory.

## Training on your own
The training scripts are available in the `training` directory.  You can download the notebooks directly from there or open the  [Question-Answer Generation](https://colab.research.google.com/drive/15GAaD-33jw81sugeBFj_Bp9GkbE_N6E1?usp=sharing) and [Distractor Generation](https://colab.research.google.com/drive/1kWZviQVx1BbelWp0rwZX7H3GIPS7_ZrP?usp=sharing) in Google Colab. 
