from typing import List, Dict
import tqdm.notebook as tq
import json
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

MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 64
TARGET_MAX_TOKEN_LEN = 24
LEARNING_RATE = 0.0001

class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

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


class AnswerGenerator():
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

        checkpoint_path = 'app/ml_models/answer_generation/models/squad-answer-generation.ckpt'
        self.ag_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.ag_model.freeze()
        self.ag_model.eval()

    def generate(self, context: str, generate_count: int) -> List[str]:
        model_output = self._model_predict(context, generate_count)

        answers = model_output.replace('<pad>', '').split('</s>')[:-1]

        return answers

    def _model_predict(self, context: str, generate_count: int) -> str:
        source_encoding = self.tokenizer(
            context,
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids = self.ag_model.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=generate_count,
            num_return_sequences=generate_count,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)

