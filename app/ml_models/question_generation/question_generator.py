from logging import NullHandler
from typing import List, Dict, Tuple
import tqdm.notebook as tq
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
    )

# Constants
MODEL_NAME = 't5-small'
LEARNING_RATE = 0.0001
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = '<sep>'
TOKENIZER_LEN = 32101 #after adding the new <sep> token

# QG Model
class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(TOKENIZER_LEN) #resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
  
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


class QuestionGenerator():
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        # print('tokenizer len before: ', len(self.tokenizer))
        self.tokenizer.add_tokens(SEP_TOKEN)
        # print('tokenizer len after: ', len(self.tokenizer))
        self.tokenizer_len = len(self.tokenizer)

        checkpoint_path = 'app/ml_models/question_generation/models/multitask-qg-ag.ckpt'
        self.qg_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.qg_model.freeze()
        self.qg_model.eval()

    def generate(self, answer: str, context: str) -> str:
        model_output = self._model_predict(answer, context)

        generated_answer, generated_question = model_output.split('<sep>')

        return generated_question

    def generate_qna(self, context: str) -> Tuple[str, str]:
        answer_mask = '[MASK]'
        model_output = self._model_predict(answer_mask, context)

        qna_pair = model_output.split('<sep>')

        if len(qna_pair) < 2:
            generated_answer = ''
            generated_question = qna_pair[0]
        else:
            generated_answer = qna_pair[0]
            generated_question = qna_pair[1]

        return generated_answer, generated_question

    def _model_predict(self, answer: str, context: str) -> str:
        source_encoding = self.tokenizer(
            '{} {} {}'.format(answer, SEP_TOKEN, context),
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids = self.qg_model.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=16,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)
