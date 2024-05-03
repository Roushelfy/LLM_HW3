from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AdamW


def show_valid_length():     
    with open('dataset/valid.json', 'r', encoding='utf-8') as f:
        valid_data = json.load(f)

    lengths = [len(x["text"]) for x in valid_data]
    plt.hist(lengths, bins=20, edgecolor='black')  # bins参数可以根据数据调整，用于控制柱状图的分组数量
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig("show.png")
    
def load_dataset(source, load_path):
    try:
        dataset = load_from_disk(load_path)
        print("load from disk")
    except:
        os.environ['http_proxy'] = "http://127.0.0.1:1080"
        os.environ['https_proxy'] = "http://127.0.0.1:1080"
        dataset = load_dataset(*source)
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        dataset.save_to_disk(load_path)
        print("load from huggingface and save")
    return dataset

def preprocess_for_lm(examples):
    inputs = tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = inputs.input_ids.copy()
    if tokenizer.eos_token_id in inputs["labels"]:
        inputs["labels"] = [[label for label in seq if label != tokenizer.eos_token_id] for seq in inputs["labels"]]
    else:
        inputs["labels"] = [seq[:-1] for seq in inputs["labels"]] 
        inputs["labels"] = [[-100] + seq for seq in inputs["labels"]]  
    
    return inputs


if __name__ == "__main__":
    
    # Setup device (CUDA if available, else CPU)
    torch.cuda.set_device(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("opt-350m")
    model = AutoModelForCausalLM.from_pretrained("opt-350m", torch_dtype=torch.float32).to(device) # 将模型转移到GPU
    
    with open('dataset/valid.json', 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    dataset = load_dataset(["RealTimeData/bbc_news_alltime", "2024-03"], "bbc_news")
    dataset = dataset.map(preprocess_for_lm, batched=True)
    # 假设batch_size为16，根据您的硬件配置调整
    batch_size = 8
    from dataclasses import asdict
    sample = dataset["train"][0]
    print(sample)

    def collate_fn(batch):
        numerical_features = ["input_ids", "attention_mask", "labels"]
        batch_dict = {k: [d[k] for d in batch] for k in numerical_features}
        
        # 将数值型特征转换为torch.tensor
        for key in numerical_features:
            batch_dict[key] = torch.tensor(batch_dict[key], dtype=torch.long)
        
        return batch_dict

    # 转换数据集为PyTorch的DataLoader
    train_dataloader = DataLoader(
        dataset=dataset["train"],
        shuffle=True,  # 训练时通常需要打乱数据
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    optimizer = AdamW(model.parameters(), lr=1e-5)  # 选择AdamW优化器，学习率可根据实际情况调整

    num_epochs = 3  # Fine-Tuning的轮数

    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # 计算损失
            loss = outputs.loss

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.save_pretrained(f"test_save{epoch}", safe_serialization=False)
        tokenizer.save_pretrained(f"test_save{epoch}", safe_serialization=False)