from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score,roc_curve

def calculatePerplexity(sentence, model, tokenizer,device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
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

def min_k_prob(all_prob, ratio):
    k_length = int(len(all_prob)*ratio)
    topk_prob = np.sort(all_prob)[:k_length]
    return np.mean(topk_prob).item()

if __name__ == "__main__":
     # Setup device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("detected_model")
    model = AutoModelForCausalLM.from_pretrained("detected_model", torch_dtype=torch.bfloat16).to(device)
    # Load data from valid.json
    with open('dataset/myvalid.json', 'r', encoding='utf-8') as f:
        valid_data = json.load(f)

    perplexities = []
    labels = []
    all_probs = []
    all_losses = []
    # Calculate perplexity
    for entry in tqdm(valid_data, desc="Calculating perplexities"):
        text = entry["text"]
        label = entry["label"]
        ppx, all_prob, mean_prob,loss = calculatePerplexity(text, model, tokenizer, device)
        all_probs.append(all_prob)
        all_losses.append(loss)
        perplexities.append(ppx)
        labels.append(1 if label == "dirty" else 0)  # Convert labels to binary format

    ratio=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # Calculate min_k_prob for each example (optional, adjust as needed)
    min_k=[]
    for r in ratio:
        min_k.append([min_k_prob(all_prob, r) for all_prob in all_probs])
    
    #Calculate AUC for each k
    for r, min_k_value in zip(ratio, min_k):
        auc_k = roc_auc_score(labels, min_k_value)
        print(f"K: {r}, AUC: {auc_k}")
    
    # Calculate AUC for loss
    auc_loss = roc_auc_score(labels, all_losses)
    print(f"Loss AUC: {auc_loss}")

    fpr, tpr, thresholds = roc_curve(labels, all_losses)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]
    print(f"Optimal Threshold: {optimal_threshold}")
    predictions = [1 if loss > optimal_threshold else 0 for loss in all_losses]

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy}")