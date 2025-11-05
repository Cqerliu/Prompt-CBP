import os
import argparse
import sys
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer
from DataSet.MyDataset import *
from model.BertCBP import BertCBP
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, 
                           matthews_corrcoef, confusion_matrix, classification_report)
import torch.nn.functional as F


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CBP模型预测脚本')
    parser.add_argument('--model_path', type=str, 
                       default='E:/pythonProject/BERT-CBP/checkpoints/CBP/model_B.subtilis.pth',
                       help='模型检查点路径')
    parser.add_argument('--test_data', type=str,
                       default='E:/pythonProject/BERT-CBP/DataSet/data/full_data/data_E.coli.csv',
                       help='测试数据路径')
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3, 4],
                       help='模型模式: 1=全模型, 2=prefix tuning, 3=adapter, 4=混合')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--pre_seq_len', type=int, default=6,
                       help='prefix序列长度')
    parser.add_argument('--output_file', type=str, default="prediction.json",
                       help='结果保存文件路径（默认JSON格式）')
    return parser.parse_args()


def load_model_and_config(args):
    """加载模型和配置"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(r"E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
    config.model_name_or_path = r"E:\pythonProject\BERT-CBP\DNABERT-2-117M"
    config.pre_seq_len = args.pre_seq_len
    config.hidden_dropout_prob = 0.2
    config.mode = args.mode
    
    # 创建模型
    bert_cbp = BertCBP(config)
    
    # 加载检查点
    if os.path.exists(args.model_path):
        print(f"加载模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        bert_cbp.load_state_dict(checkpoint)
    else:
        print(f"警告: 模型文件不存在: {args.model_path}")
        print("使用随机初始化的模型")
    
    bert_cbp.to(device)
    bert_cbp.eval()
    
    return bert_cbp, device

def load_test_data(args):
    """加载测试数据"""
    print(f"加载测试数据: {args.test_data}")
    dataset = pd.read_csv(args.test_data)
    
    test_sentences = dataset["sequence"]
    test_labels = dataset["label"]
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(r"E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
    
    # 转换为token并添加padding
    test_inputs, test_labels = input_token_test(test_sentences, test_labels, tokenizer)
    
    # 创建数据集和数据加载器
    test_dataset = MyDataset(test_inputs, test_labels)
    testloader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return testloader, len(test_labels)


def predict_and_evaluate(model, testloader, device, args):
    """进行预测并评估模型性能"""
    print(f"开始预测 (模式 {args.mode})...")
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            batch = tuple(p.to(device) for p in batch)
            input_ids, labels = batch[0], batch[1]
            
            # 进行预测
            logits = model(input_ids=input_ids)
            
            # 计算概率
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
            all_labels.extend(labels.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(testloader)} 个批次")
    
    return all_predictions, all_probabilities, all_labels


def calculate_metrics(predictions, probabilities, labels):
    """计算各种评估指标"""
    # 基本指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    auc = roc_auc_score(labels, probabilities)
    mcc = matthews_corrcoef(labels, predictions)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # 额外指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def print_results(metrics, args):
    """打印结果"""
    print("\n" + "="*60)
    print(f"模型预测结果 (模式 {args.mode})")
    print("="*60)
    print(f"准确率 (Accuracy):     {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision):    {metrics['precision']:.4f}")
    print(f"召回率 (Recall):       {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):     {metrics['f1']:.4f}")
    print(f"AUC:                   {metrics['auc']:.4f}")
    print(f"MCC:                   {metrics['mcc']:.4f}")
    print(f"敏感性 (Sensitivity):  {metrics['sensitivity']:.4f}")
    print(f"特异性 (Specificity):  {metrics['specificity']:.4f}")
    
    print(f"\n混淆矩阵:")
    print(f"真阳性 (TP): {metrics['tp']}")
    print(f"真阴性 (TN): {metrics['tn']}")
    print(f"假阳性 (FP): {metrics['fp']}")
    print(f"假阴性 (FN): {metrics['fn']}")
    
    print(f"\n混淆矩阵 (数值):")
    print(metrics['confusion_matrix'])
    print("="*60)


def save_results(metrics, args, predictions, probabilities, labels):
    """保存结果到JSON文件"""
    if args.output_file:
        # 准备结果数据
        results = {
            'model_path': args.model_path,
            'test_data': args.test_data,
            'mode': args.mode,
            'batch_size': args.batch_size,
            'pre_seq_len': args.pre_seq_len,
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'auc': float(metrics['auc']),
                'mcc': float(metrics['mcc']),
                'sensitivity': float(metrics['sensitivity']),
                'specificity': float(metrics['specificity']),
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'tp': int(metrics['tp']),
                'tn': int(metrics['tn']),
                'fp': int(metrics['fp']),
                'fn': int(metrics['fn'])
            },
            'predictions': [int(x) for x in predictions] if isinstance(predictions, np.ndarray) else [int(x) for x in predictions],
            'probabilities': [float(x) for x in probabilities] if isinstance(probabilities, np.ndarray) else [float(x) for x in probabilities],
            'labels': [int(x) for x in labels] if isinstance(labels, np.ndarray) else [int(x) for x in labels]
        }
        
        # 确保文件扩展名为.json
        json_file = args.output_file
        if not json_file.endswith('.json'):
            json_file = json_file + '.json'
        
        # 保存为JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON结果已保存到: {json_file}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("CBP模型预测脚本")
    print(f"模型路径: {args.model_path}")
    print(f"测试数据: {args.test_data}")
    print(f"模型模式: {args.mode}")
    print(f"批处理大小: {args.batch_size}")
    print(f"Prefix长度: {args.pre_seq_len}")
    
    # 加载模型
    model, device = load_model_and_config(args)
    
    # 加载测试数据
    testloader, num_samples = load_test_data(args)
    print(f"测试样本数量: {num_samples}")
    
    # 进行预测
    predictions, probabilities, labels = predict_and_evaluate(model, testloader, device, args)
    
    # 计算指标
    metrics = calculate_metrics(predictions, probabilities, labels)
    
    # 打印结果
    print_results(metrics, args)
    
    # 保存结果
    if args.output_file:
        save_results(metrics, args, predictions, probabilities, labels)


if __name__ == "__main__":
    main()

