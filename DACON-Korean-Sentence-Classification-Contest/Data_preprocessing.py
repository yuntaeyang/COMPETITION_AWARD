
import pandas as pd
import numpy as np

path = "/content/drive/MyDrive/한국어 문장 관계 분류 경진대회/open/"

train_1 = pd.read_csv(f"{path}train_data.csv")
train_2 = pd.read_csv(f"{path}plus_data.csv")
test = pd.read_csv(f"{path}test_data.csv")

train=pd.concat([train_1, train_2])

list1 = [(train['label']== "entailment"), (train['label']== "contradiction"), (train['label']== "neutral")]
choicelist1 = [0,1,2]
train['label']=np.select(list1, choicelist1)

train=train[['premise','hypothesis','label']]
test=test[['premise','hypothesis']]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)