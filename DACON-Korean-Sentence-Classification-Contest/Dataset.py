import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

class TRAINDataset(Dataset):
  
    def __init__(self, data):
        self.dataset = data
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        print(self.dataset)
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:3].values
        sentence1 = row[0]
        sentence2 = row[1]
        y = row[2]
        inputs = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            add_special_tokens=True,
            max_length=100
        )
    
        input_ids = torch.from_numpy(np.asarray(inputs['input_ids']))
        attention_mask = torch.from_numpy(np.asarray(inputs['attention_mask']))

        return input_ids, attention_mask, y

class TESTDataset(Dataset):
  
    def __init__(self, data):
        self.dataset = data
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        print(self.dataset)
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:2].values
        sentence1 = row[0]
        sentence2 = row[1]
        inputs = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            add_special_tokens=True,
            max_length=100
            )
    
        input_ids = torch.from_numpy(np.asarray(inputs['input_ids']))
        attention_mask = torch.from_numpy(np.asarray(inputs['attention_mask']))

        return input_ids, attention_mask