# This file is used for cross species promoter prediction

import math
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer
from DataSet.MyDataset import *
from model.BertCBP import BertCBP
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)
import torch.nn.functional as F

# --------------------parameter configuration----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'E:/pythonProject/BERT-CBP/checkpoints/CBP/model_E.coli.pth'
save_prompt = "E:/pythonProject/BERT-CBP/checkpoints/prompt/prompt_model_E.coli2B.subtilis.pth"
dataset_folder = 'E:/pythonProject/BERT-CBP/DataSet/data/test/test_data_B.subtilis.csv'
config = AutoConfig.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
config.model_name_or_path = "E:\pythonProject\BERT-CBP\DNABERT-2-117M"
config.pre_seq_len = 6
config.hidden_dropout_prob = 0.2
full_model = False
config.full_model = full_model
best_test_accuracy = 0.0

dataset = pd.read_csv(dataset_folder)
# Test_dataset
test_sentences = dataset["sequence"]
test_labels = dataset["label"]

tokenizer = AutoTokenizer.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
# Transformed into token and add padding
test_inputs, test_labels = input_token_test(test_sentences, test_labels, tokenizer)

# Calculate the length after adding padding
test_dataset = MyDataset(test_inputs, test_labels)
testloader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Loading model to predict
if full_model:
    bert_cbp = BertCBP(config)
    bert_cbp.load_state_dict(torch.load(save_path))
    bert_cbp.to(device)
else:
    bert_cbp = BertCBP(config)
    bert_cbp.load_state_dict(torch.load(save_prompt))
    bert_cbp.to(device)
print('Starting predict')
correct, total, pos_num, tp = 0, 0, 0, 0
preds = []
probs = []
labels = []
bert_cbp.eval()
with torch.no_grad():
    for i, batch in enumerate(testloader):
        batch = tuple(p.to(device) for p in batch)
        pred = bert_cbp(batch[0])
        prob = F.softmax(pred, dim=1)
        preds.extend(prob[:, 1].cpu().detach().numpy().tolist())
        labels.extend(batch[1].cpu().detach().numpy().tolist())
        _, predicted = torch.max(pred, 1)
        total += batch[1].size(0)
        correct += (predicted == batch[1]).sum().item()
        pos_num += (batch[1] == 1).sum().item()
        tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
    neg_num = total - pos_num
    tn = correct - tp
    sn = tp / pos_num if pos_num != 0 else 1
    sp = tn / neg_num if neg_num != 0 else 1
    acc = (tp + tn) / (pos_num + neg_num) if (pos_num + neg_num) != 0 else 1
    fn = pos_num - tp
    fp = neg_num - tn
    mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) \
        if (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) != 0 else 1
    auc = roc_auc_score(labels, preds)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 1
    recall = tp / pos_num if pos_num != 0 else 1
    precision = tp / (tp + fp) if (tp + fp) != 0 else 1
    print('Acc = %.4f  f1=%.4f  Sn = %.4f  Sp = %.4f  precision=%.4f  recall=%.4f  auc=%.4f  Mcc= %.4f' % (
        acc, f1, sn, sp, precision, recall, auc, mcc))
    print('--------------------------------')

