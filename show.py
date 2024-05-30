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


if __name__ == "__main__":
    