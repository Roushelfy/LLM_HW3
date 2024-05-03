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
import random


def calculatePerplexity(sentence, model, tokenizer,device):
    inputs = tokenizer(sentence, max_length=2048, padding="longest", truncation=True, return_tensors="pt").to(device)
    input_ids=inputs.input_ids
    with torch.no_grad():
        output = model(**inputs,labels=input_ids)
        logits = output.logits
        loss = output.loss

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob), -loss.item()


if __name__ == "__main__":
     # Setup device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")
    with open('dataset/valid.json', 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    # models = ["opt-350m", "detected_model", "results2/final"]
    # for i in range(1, 10):
    #     models.append(f"results/checkpoint-{i*10}")
    # for i in range(1, 17):
    #     models.append(f"results2/checkpoint-{i*10}")
    models = ["opt-350m"]
    
    for model_path in models:
        tokenizer = AutoTokenizer.from_pretrained("results2/final")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        # Load data from valid.json

        perplexities = []
        labels = []
        all_probs = []
        all_losses = []
        # Calculate perplexity
        for entry in tqdm(valid_data, desc="Calculating perplexities"):
            text = entry["text"]
            label = "dirty"
            # label = entry["label"]
            if random.random() > 0.5:
                continue
            ppx, all_prob, mean_prob,loss = calculatePerplexity(text, model, tokenizer, device)
            all_probs.append(all_prob)
            all_losses.append(loss)
            perplexities.append(ppx)
            labels.append(1 if label == "dirty" else 0)  # Convert labels to binary format
        print(model_path, "valid", sum(perplexities) / len(perplexities))
        
        # perplexities = []
        # labels = []
        # all_probs = []
        # all_losses = []
        # for entry in tqdm(dataset, desc="Calculating perplexities"):
        #     text = entry["content"]
        #     # print(text)
        #     ppx, all_prob, mean_prob,loss = calculatePerplexity(text, model, tokenizer, device)
        #     all_probs.append(all_prob)
        #     all_losses.append(loss)
        #     perplexities.append(ppx)
        # print(model_path, "collected", sum(perplexities) / len(perplexities))
        