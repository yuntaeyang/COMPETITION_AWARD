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
    parser.add_argument('--batch_size', default=16, type=int, help='batch for training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = torch.device('cuda')
    return args

path = "/content/drive/MyDrive/한국어 문장 관계 분류 경진대회/open/"
test=pd.read_csv(f"{path}test.csv")
submission = pd.read_csv(f"{path}sample_submission.csv")

# 예측 
def inference(model, dataset_test, batch_size, device):
    test_dataset = TESTDataset(dataset_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    output_pred = []
    with torch.no_grad():
      for batch_id, (token_ids, attention_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
        token_ids = token_ids.long().to(device)
        attention_masks = attention_masks.long().to(device)
        output=model(token_ids, attention_masks)[0]
        logits = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        output_pred.extend(logits)
    return output_pred

label_dict = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}

# 결과 도출
def inference_main(args):
  res = np.zeros((len(test),3)) 
  for i in range(5): 
    print(f'fold{i} 모델 추론중...')
    # load my model
    model = torch.load(str(path)+'model'+str(i)+'.pt')

    pred_answer = inference(model, test, args.batch_size, args.device)

    res += np.array(pred_answer) / 5 

  ans= np.argmax(res, axis=-1)
  out = [list(label_dict.keys())[_] for _ in ans]
  submission["label"] = out

  submission.to_csv("FOLD5(10)_submission.csv", index = False)

if __name__ == "__main__":
    args = parse_args()
    inference_main(args)
