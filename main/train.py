# Four training strategies are supported:

# 1.Used for training baseline models.
# 2.Prompt strategy.
# 3.Output Fine-tuning strategy.
# 4.Prompt+Output Fine tuning strategy.
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import math
import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from torch.optim import AdamW
from DataSet.MyDataset import *
from adjust_learning import *
from model.BertCBP import BertCBP
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import KFold
import json
import statistics

if __name__ == '__main__':
    # Set Random Seed
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ----------------------参数配置-------------------------
    # Model Path

    # 根据训练的数据集动态设置模型路径
    # model_path = 'E:/pythonProject/BERT-CBP/checkpoints/CBP/model_E.coli.pth'
    # save_prompt = "E:/pythonProject/BERT-CBP/checkpoints/prompt/prompt_model_E.coli2B.subtilis.pth"
    # Dataset Path (use full_data directly)
    full_data_path = 'E:/pythonProject/BERT-CBP/DataSet/data/full_data/data_D.melanogaster.csv'
    # training parameters
    epoches = 30
    # 根据DNABERT-2文档示例，先加载配置再创建模型
    config = AutoConfig.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
    config.model_name_or_path = "E:\pythonProject\BERT-CBP\DNABERT-2-117M"
    # 添加自定义配置参数
    config.pre_seq_len = 6
    config.hidden_dropout_prob = 0.2
    mode = 2
    lora = False
    config.mode = mode
    ratio = 1
    results_save_path = "E:/pythonProject/BERT-CBP/checkpoints/results/cv_results.json"

    # Load Dataset and prepare for 5-fold CV (directly from full_data)
    df_all = pd.read_csv(full_data_path)
    # parse species from file name, e.g., data_E.coli.csv -> E.coli
    species = os.path.splitext(os.path.basename(full_data_path))[0]
    if species.startswith('data_'):
        species = species[len('data_'):]
    
    # 当mode=1时，设置保存路径
    if mode == 1:
        save_path = f"E:/pythonProject/BERT-CBP/checkpoints/CBP/model_{species}.pth"
        model_path = None  # mode=1不需要预训练模型
        model_species = ""
        cross_species = ""
    else:
        # 对于mode 2,3,4，需要预训练模型
        # 默认使用E.coli模型作为预训练模型，或者可以根据需要修改
        model_species = "B.subtilis"  # 可以修改为其他物种
        model_path = f"E:/pythonProject/BERT-CBP/checkpoints/CBP/model_{model_species}.pth"
        cross_species = f"{model_species}2{species}"
        
        # 检查预训练模型是否存在
        if not os.path.exists(model_path):
            print(f"警告: 预训练模型不存在: {model_path}")
            print("请确保已训练好对应的基线模型，或修改model_species变量")
            exit(1)
    if ratio != 0:
        df_all = df_all.sample(frac=ratio, random_state=1).reset_index(drop=True)
    # 打印df_all前20个样本
    # print(df_all.head(20))
    print("full_dataset:", df_all.shape)
    tokenizer = AutoTokenizer.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=41)
    cv_results = []
    total_size = len(df_all)

    # Lora
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["pooler"],
    )

    # 5-Fold Cross Validation training
    fold_index = 0
    for train_index, test_index in kfold.split(df_all):
        fold_index += 1
        print(f"========== Fold {fold_index}/5 ==========")
        train_split = df_all.iloc[train_index].reset_index(drop=True)
        test_split = df_all.iloc[test_index].reset_index(drop=True)
        # print(train_split.head(20))
        # print(test_split.head(20))
        train_size = int(len(train_index))
        test_size = int(len(test_index))

        # Build dataloaders for this fold
        train_sentences = train_split["sequence"]
        train_labels_series = train_split["label"]
        test_sentences = test_split["sequence"]
        test_labels_series = test_split["label"]
        train_inputs, train_labels_np, test_inputs, test_labels_np = input_token(
            train_sentences, train_labels_series, test_sentences, test_labels_series, tokenizer
        )
        train_dataset_fold = MyDataset(train_inputs, train_labels_np)
        test_dataset_fold = MyDataset(test_inputs, test_labels_np)
        trainloader = Data.DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
        testloader = Data.DataLoader(test_dataset_fold, batch_size=32, shuffle=False)

        # Initialize model per fold according to mode
        if mode == 1:
            bert_cbp = BertCBP(config)
            bert_cbp.to(device)
            if lora:
                bert_cbp.bert = get_peft_model(bert_cbp.bert, lora_config)
                bert_cbp.bert.print_trainable_parameters()
                optimizer = AdamW([
                    {"params": bert_cbp.bert.parameters()},
                    {"params": bert_cbp.conv1d.parameters()},
                    {"params": bert_cbp.lstm.parameters()},
                    {"params": bert_cbp.classifier.parameters()}
                ], lr=1.5e-5, weight_decay=1e-2)
            else:
                optimizer = AdamW(bert_cbp.parameters(), lr=1.5e-5, weight_decay=1e-2)
        elif mode == 2:
            bert_cbp = BertCBP(config)
            bert_cbp.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            bert_cbp.to(device)
            
            # 冻结BERT参数
            for param in bert_cbp.bert.parameters():
                param.requires_grad = False
            
            # 冻结下游任务参数（CNN, LSTM, BN, classifier）
            for param in bert_cbp.conv1d.parameters():
                param.requires_grad = False
            for param in bert_cbp.lstm.parameters():
                param.requires_grad = False
            for param in bert_cbp.bn.parameters():
                param.requires_grad = False
            for param in bert_cbp.classifier.parameters():
                param.requires_grad = False
            
            # 解冻prefix_encoder参数
            for param in bert_cbp.prefix_encoder.parameters():
                param.requires_grad = True
            
            # 收集需要更新的参数
            param_to_update = []
            for name, param in bert_cbp.named_parameters():
                if param.requires_grad:
                    param_to_update.append(param)
                    print(f"可训练参数: {name}, 形状: {param.shape}")
            
            if len(param_to_update) == 0:
                print("错误: 没有找到可训练的参数!")
                exit(1)
                
            optimizer = AdamW(param_to_update, lr=1.5e-4, weight_decay=1e-2)
            print(f"优化器包含 {len(param_to_update)} 个参数组")
            print("Mode 2: 训练prefix_encoder参数，其他参数保持冻结")
        elif mode == 3:
            bert_cbp = BertCBP(config)
            bert_cbp.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            bert_cbp.to(device)
            
            # 冻结BERT参数
            for param in bert_cbp.bert.parameters():
                param.requires_grad = False
            
            # 冻结prefix_encoder和下游任务参数（CNN, LSTM, BN）
            for param in bert_cbp.conv1d.parameters():
                param.requires_grad = False
            for param in bert_cbp.lstm.parameters():
                param.requires_grad = False
            for param in bert_cbp.bn.parameters():
                param.requires_grad = False
            for param in bert_cbp.prefix_encoder.parameters():
                param.requires_grad = False
            # 解冻分类器参数
            for param in bert_cbp.classifier.parameters():
                param.requires_grad = True
            
            # 收集需要更新的参数
            param_to_update = []
            for name, param in bert_cbp.named_parameters():
                if param.requires_grad:
                    param_to_update.append(param)
                    print(f"可训练参数: {name}, 形状: {param.shape}")
            
            if len(param_to_update) == 0:
                print("错误: 没有找到可训练的参数!")
                exit(1)
                
            optimizer = AdamW(param_to_update, lr=1.5e-5, weight_decay=1e-2)
            print(f"优化器包含 {len(param_to_update)} 个参数组")
            print("Mode 3: 训练分类器参数，其他参数保持冻结")
        elif mode == 4:
            bert_cbp = BertCBP(config)
            bert_cbp.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            bert_cbp.to(device)
            
            # 冻结BERT参数
            for param in bert_cbp.bert.parameters():
                param.requires_grad = False
            # 冻结下游任务相关参数（CNN, LSTM, Classifier）
            for param in bert_cbp.conv1d.parameters():
                param.requires_grad = False
            for param in bert_cbp.lstm.parameters():
                param.requires_grad = False
            for param in bert_cbp.bn.parameters():
                param.requires_grad = False
            # 解冻prefix_encoder和classifier参数
            for param in bert_cbp.prefix_encoder.parameters():
                param.requires_grad = True
            for param in bert_cbp.classifier.parameters():
                param.requires_grad = True
            
            # 收集需要更新的参数
            param_to_update = []
            for name, param in bert_cbp.named_parameters():
                if param.requires_grad:
                    param_to_update.append(param)
                    print(f"可训练参数: {name}, 形状: {param.shape}")
            
            if len(param_to_update) == 0:
                print("错误: 没有找到可训练的参数!")
                exit(1)
                
            optimizer = AdamW(param_to_update, lr=1.5e-5, weight_decay=1e-2)
            print(f"优化器包含 {len(param_to_update)} 个参数组")
            print("Mode 4: 训练prefix_encoder和分类器参数，其他参数保持冻结")

        # Count trainable parameters for this fold
        trainable_params = sum(p.numel() for p in bert_cbp.parameters() if p.requires_grad)

        loss_fn = nn.CrossEntropyLoss()
        best_metrics = {"acc": 0.0, "f1": 0.0, "sn": 0.0, "sp": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0, "mcc": 0.0}
        best_epoch = -1

        for epoch in tqdm(range(0, epoches)):
            bert_cbp.train()
            # corrcet_number, total_number, real positive number, real and predict both are positive number
            correct, total, pos_num, tp = 0, 0, 0, 0
            preds = []
            labels = []
            for i, batch in enumerate(trainloader):
                optimizer.zero_grad()
                batch = tuple(p.to(device) for p in batch)
                pred = bert_cbp(batch[0])
                loss = loss_fn(pred, batch[1])
                loss.backward()
                # adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=epoches, lr_min=2e-6,
                #                      lr_max=1.5e-5,
                #                      warmup=True)
                optimizer.step()
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
            print('Train: Acc = %.4f  f1=%.4f  Sn = %.4f  Sp = %.4f  precision=%.4f  recall=%.4f  auc=%.4f  Mcc= %.4f' % (
                acc, f1, sn, sp, precision, recall, auc, mcc))

            # Evaluation on fold validation
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
                print('Val  : Acc = %.4f  f1=%.4f  Sn = %.4f  Sp = %.4f  precision=%.4f  recall=%.4f  auc=%.4f  Mcc= %.4f' % (
                acc, f1, sn, sp, precision, recall, auc, mcc))

                # Track best metrics by accuracy (or auc)
                if acc > best_metrics["acc"]:
                    best_metrics = {"acc": acc, "f1": f1, "sn": sn, "sp": sp, "precision": precision, "recall": recall, "auc": auc, "mcc": mcc}
                    best_epoch = epoch
                    # 当mode=1时，保存最佳模型
                    if mode == 1:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(bert_cbp.state_dict(), save_path)

        cv_results.append({
            "fold": fold_index,
            "mode": mode,
            "trainable_params": int(trainable_params),
            "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else -1,
            "metrics": best_metrics,
            "data_sizes": {"total": int(total_size), "train": train_size, "test": test_size}
        })

    # Save CV results as proper JSON array format
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    
    # Calculate average metrics and standard deviations
    avg_metrics = {}
    std_metrics = {}
    
    for metric in cv_results[0]["metrics"].keys():
        values = [d["metrics"][metric] for d in cv_results]
        avg_metrics[metric] = sum(values) / len(values)
        std_metrics[metric] = statistics.stdev(values) if len(values) > 1 else 0.0
    
    run_result = {
        "mode": mode,
        "species": species,
        "model_species": model_species,
        "cross_species": cross_species,
        "total_size": int(total_size),
        "folds": cv_results,
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics
    }
    
    # Load existing results if file exists
    existing_results = []
    if os.path.exists(results_save_path):
        try:
            with open(results_save_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []
    
    # Add new result to existing results
    existing_results.append(run_result)
    
    # Save updated results as proper JSON array
    with open(results_save_path, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    print(f"CV results saved to: {results_save_path}")
    
    # 当mode=1时，保存最终的最佳模型
    if mode == 1:
        print(f"训练完成！最佳模型已保存到: {save_path}")

