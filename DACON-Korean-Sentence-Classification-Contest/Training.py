import pandas as pd
import numpy as np
import os
import argparse
import transformers
from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, tqdm_notebook
import random
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold
from adamp import AdamP
from Dataset import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training.')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch for training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = torch.device('cuda')
    return args

path = "/content/drive/MyDrive/한국어 문장 관계 분류 경진대회/open/"
train=pd.read_csv(f"{path}train.csv")


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# 모델 학습 및 검증
def training(train_dataset,val_dataset, fold, epoch, batch, device):
  best_acc = 0
  
  model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=3).to(device)
  
  dataset_train = TRAINDataset(train_dataset)
  dataset_val = TRAINDataset(val_dataset)

  train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True)
  valid_loader = DataLoader(dataset_val, batch_size=batch, shuffle=False)

  optimizer = AdamP(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2)

  total_steps = len(train_loader) * epoch

  # 스케줄러
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)

  for e in range(epoch):
    train_acc = 0.0
    valid_acc = 0.0
    model.train()
    for batch_id, (token_ids, attention_masks, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
      optimizer.zero_grad()
      token_ids = token_ids.to(device)
      attention_masks = attention_masks.to(device)
      label = label.to(device)
      out = model(token_ids, attention_masks)[0]
      loss = F.cross_entropy(out, label)
      loss.backward()
      optimizer.step()
      scheduler.step()
      train_acc += calc_accuracy(out, label)

    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval()
    for batch_id, (token_ids, attention_masks, label) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
      token_ids = token_ids.to(device)
      attention_masks = attention_masks.to(device)
      label = label.to(device)
      out = model(token_ids, attention_masks)[0]
      valid_acc += calc_accuracy(out, label)
    print("epoch {} valid acc {}".format(e+1, valid_acc / (batch_id+1)))
#    if valid_acc > best_acc:
#      torch.save(model, '/content/drive/MyDrive/한국어 문장 관계 분류 경진대회/open/model'+str(fold)+'.pt')
    torch.save(model, str(path)+'model'+str(fold)+'.pt')

# 교차검증
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    # kfold
    kfold=[]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    for train_idx, val_idx in splitter.split(train.iloc[:, 0:2],train.iloc[:, 2]):
        kfold.append((train.iloc[train_idx,:],train.iloc[val_idx,:]))

    for fold,(train_datasets, valid_datasets) in enumerate(kfold):
        print(f'fold{fold} 학습중...')
        training(train_dataset=train_datasets,val_dataset=valid_datasets,fold=fold, epoch=args.epochs, batch=args.batch_size, device=args.device)

if __name__ == "__main__":
    args = parse_args()
    main(args)