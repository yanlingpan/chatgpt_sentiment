import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification

from parse import replace_hash, strip_hashtag_sequence


# Dataset
class TweetDataset(torch.utils.data.Dataset):
  def __init__(self, text_list, text_labels=None, preprocess=None):
    self.text_labels = text_labels
    self.text_list = text_list
    self.preprocess = preprocess

  def __len__(self):
    return len(self.text_list)

  def __getitem__(self, idx):
    text = self.text_list.iloc[idx]
    if self.preprocess:
      for process in self.preprocess:
        text = process(text)
    if self.text_labels is not None:
      label = self.text_labels.iloc[idx]
      return text, label
    else:
      return text


# model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512, use_fast=True)
config = AutoConfig.from_pretrained(MODEL) # used for id to label name
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)

# data
data_dir = "data/"
data = pd.read_csv(Path(data_dir, "tweets172clean.csv"), index_col=0)
dataset = TweetDataset(data.text, preprocess=[strip_hashtag_sequence, replace_hash, ])
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# predict
model.eval()
with torch.no_grad():
  y_pred, y_prob, y_logits = [], [], []
  for batch, texts in enumerate(dataloader):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask) # logits only
    logits = outputs[0]

    y_pred.append(torch.argmax(logits, dim=1))
    y_prob.append(torch.softmax(logits, dim=1))
    y_logits.append(logits)

  y_pred = torch.cat(y_pred).cpu() - 1 # cast as [-1, 0, 1] sentiment
  y_prob = torch.cat(y_prob).cpu()
  y_logits = torch.cat(y_logits).cpu()
  assert len(y_pred) == len(y_prob) == len(y_logits)

np.save(Path(data_dir, 'xlm_pred.npy'), y_pred.numpy())
np.save(Path(data_dir, 'xlm_prob.npy'), y_prob.numpy())
np.save(Path(data_dir, 'xlm_logits.npy'), y_logits.numpy())
