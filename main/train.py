# This file is used to train the Prompt-CBP model. Here, mode has a total of 4 values.
# They are respectively represented as follows.
# 1: Used for training baseline models.
# 2: Prompt strategy.
# 3: Output Fine-tuning strategy.
# 4: Prompt+Output Fine tuning strategy.

import math
import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn
from transformers import AdamW, AutoConfig, AutoTokenizer
from DataSet.MyDataset import *
from adjust_learning import *
from model.BertCBP import BertCBP
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

if __name__ == '__main__':
    # Set Random Seed
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----------------------parameter configuration-------------------------
    # Model Path
    save_path = "E:/pythonProject/BERT-CBP/checkpoints/CBP/model_D.melanogaster.pth"

    model_path = 'E:/pythonProject/BERT-CBP/checkpoints/CBP/model_E.coli.pth'
    save_prompt = "E:/pythonProject/BERT-CBP/checkpoints/prompt/prompt_model_E.coli2B.subtilis.pth"
    # Dataset Path
    train_dataset_path = 'E:/pythonProject/BERT-CBP/DataSet/data/train/train_data_B.subtilis.csv'
    test_dataset_path = 'E:/pythonProject/BERT-CBP/DataSet/data/test/test_data_B.subtilis.csv'
    # training parameters
    epoches = 50
    config = AutoConfig.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
    config.model_name_or_path = "E:\pythonProject\BERT-CBP\DNABERT-2-117M"
    config.pre_seq_len = 6
    config.hidden_dropout_prob = 0.2

    mode = 4
    lora = False
    config.mode = mode
    best_test_accuracy = 0.0
    ratio = 0.1

    # Load Dataset
    train_dataset = pd.read_csv(train_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)
    if ratio != 0:
        dataset = pd.concat([train_dataset, test_dataset])
        train_dataset = dataset.sample(frac=ratio, random_state=1)
        train_dataset.reset_index(drop=True, inplace=True)
        test_dataset = dataset.drop(train_dataset.index)
        test_dataset.reset_index(drop=True, inplace=True)
    print("train_dataset:", train_dataset.shape)
    print("test_dataset:", test_dataset.shape)
    # Train_dataset
    train_sentences = train_dataset["sequence"]
    train_labels = train_dataset["label"]
    # Test_dataset
    test_sentences = test_dataset["sequence"]
    test_labels = test_dataset["label"]
    tokenizer = AutoTokenizer.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
    # Transformed into token and add padding
    train_inputs, train_labels, test_inputs, test_labels = input_token(train_sentences, train_labels,
                                                                       test_sentences, test_labels, tokenizer)

    # Calculate the length after adding padding
    train_dataset = MyDataset(train_inputs, train_labels)
    test_dataset = MyDataset(test_inputs, test_labels)
    trainloader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Lora
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["pooler"],
    )

    # Loading model
    if mode == 1:
        bert_cbp = BertCBP(config)
        bert_cbp.to(device)
        if lora:
            bert_cbp.bert = get_peft_model(bert_cbp.bert, lora_config)
            bert_cbp.bert.print_trainable_parameters()
            optimizer = AdamW([{"params": bert_cbp.bert.parameters()},
                               {"params": bert_cbp.conv1d.parameters()},
                               {"params": bert_cbp.lstm.parameters()},
                               {"params": bert_cbp.classifier.parameters()}
                               ], lr=1.5e-5, weight_decay=1e-2)
        else:
            optimizer = AdamW(bert_cbp.parameters(), lr=1.5e-5, weight_decay=1e-2)
    elif mode == 2:
        bert_cbp = BertCBP(config)
        bert_cbp.load_state_dict(torch.load(model_path))
        bert_cbp.to(device)
        for param in bert_cbp.parameters():
            param.requires_grad = False
        for param in bert_cbp.prefix_encoder.parameters():
            param.requires_grad = True
        param_to_update = []
        for param in bert_cbp.parameters():
            if param.requires_grad:
                param_to_update.append(param)
        # param_update = bert_cbp.prefix_encoder.parameters()

        # Select optimizer and loss function
        optimizer = AdamW(param_to_update, lr=1.5e-5, weight_decay=1e-2)
    elif mode == 3:
        bert_cbp = BertCBP(config)
        bert_cbp.load_state_dict(torch.load(model_path))
        bert_cbp.to(device)
        param_update = bert_cbp.classifier.parameters()
        optimizer = AdamW(param_update, lr=1.5e-5, weight_decay=1e-2)
    elif mode == 4:
        bert_cbp = BertCBP(config)
        bert_cbp.load_state_dict(torch.load(model_path))
        bert_cbp.to(device)
        # Train the parameters of classifier and prefix_encoder
        for param in bert_cbp.parameters():
            param.requires_grad = False
        for param in bert_cbp.prefix_encoder.parameters():
            param.requires_grad = True
        for param in bert_cbp.classifier.parameters():
            param.requires_grad = True
        param_to_update = []
        for param in bert_cbp.parameters():
            if param.requires_grad:
                param_to_update.append(param)
        optimizer = AdamW(param_to_update, lr=1.5e-5, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in tqdm(range(0, epoches)):
        bert_cbp.train()
        print(f'Starting epoch {epoch + 1}')
        print('Starting training')
        # corrcet_number, total_number, real positive number, real and predict both are positive number
        correct, total, pos_num, tp = 0, 0, 0, 0
        preds = []
        labels = []
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            # batch[0] is token embedding; batch[1] is real label
            batch = tuple(p.to(device) for p in batch)
            # Data input to the model
            pred = bert_cbp(batch[0])
            # Calculate loss function
            loss = loss_fn(pred, batch[1])
            # Back Propagation
            loss.backward()
            # Warm-up and Learning Rate Decay
            adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=epoches, lr_min=2e-6,
                                 lr_max=1.5e-5,
                                 warmup=True)
            # Model weight update
            optimizer.step()
            # predicte label
            prob = F.softmax(pred, dim=1)
            preds.extend(prob[:, 1].cpu().detach().numpy().tolist())
            labels.extend(batch[1].cpu().detach().numpy().tolist())
            _, predicted = torch.max(pred, 1)
            total += batch[1].size(0)
            # correct number
            correct += (predicted == batch[1]).sum().item()
            # positive number
            pos_num += (batch[1] == 1).sum().item()
            tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
        neg_num = total - pos_num

        tn = correct - tp
        sn = tp / pos_num if pos_num != 0 else 1
        sp = tn / neg_num if neg_num != 0 else 1
        # Calculation accuracy
        acc = (tp + tn) / (pos_num + neg_num) if (pos_num + neg_num) != 0 else 1
        fn = pos_num - tp
        fp = neg_num - tn
        # Calculate Matthews correlation coefficient
        mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) \
            if (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) != 0 else 1
        auc = roc_auc_score(labels, preds)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 1
        recall = tp / pos_num if pos_num != 0 else 1
        precision = tp / (tp + fp) if (tp + fp) != 0 else 1
        print('Acc = %.4f  f1=%.4f  Sn = %.4f  Sp = %.4f  precision=%.4f  recall=%.4f  auc=%.4f  Mcc= %.4f' % (
            acc, f1, sn, sp, precision, recall, auc, mcc))
        print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
        print('Starting testing')
        # model valuing
        bert_cbp.eval()
        correct, total, pos_num, tp = 0, 0, 0, 0
        preds = []
        labels = []
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
            # ---------- saving best model ----------
            if mode == 1:
                test_accuracy = acc
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    torch.save(bert_cbp.state_dict(), save_path)
                    print(f"Saved new best model with accuracy: {test_accuracy:.2f}%")
            elif mode == 2:
                test_accuracy = acc
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    torch.save(bert_cbp.state_dict(), save_prompt)
                    print(f"Saved new best prompt model with accuracy: {test_accuracy:.2f}%")
            else:
                test_accuracy = acc
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_epoch = epoch
                print(f"best model with accuracy: {best_test_accuracy:.4f}%, epoch: {best_epoch + 1}")

