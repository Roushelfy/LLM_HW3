from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

def calculatePerplexity(sentence, model, tokenizer, device):
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
    return p1, all_prob

def calcmink(vec, ref_vec, ratio):
    k_length = max(int(len(vec)*ratio) - 1, 1)
    vec = np.array(vec)
    ref_vec = np.array(ref_vec)
    topk_vec = np.sort(vec)[:k_length]
    topk_ref_vec = np.sort(ref_vec)[:k_length]
    return np.mean(topk_vec) - np.mean(topk_ref_vec)

def calcmink_base(vec, ref_vec, ratio):
    k_length = max(int(len(vec)*ratio) - 1, 1)
    vec = np.array(vec)
    ref_vec = np.array(ref_vec)
    indexs = np.argpartition(vec, k_length)[:k_length]
    topk_vec = vec[indexs]
    topk_ref_vec = ref_vec[indexs]
    return np.mean(topk_vec) - np.mean(topk_ref_vec)
    
    
def calcmink_ref(vec, ref_vec, ratio):
    k_length = max(int(len(vec)*ratio) - 1, 1)
    vec = np.array(vec)
    ref_vec = np.array(ref_vec)
    indexs = np.argpartition(ref_vec, k_length)[:k_length]
    topk_vec = vec[indexs]
    topk_ref_vec = ref_vec[indexs]
    return np.mean(topk_vec) - np.mean(topk_ref_vec)
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("detected_model")
    model = AutoModelForCausalLM.from_pretrained("detected_model").to(device)
    ref_paths = ["opt-350m", "results2/final"]
    for i in range(1, 10):
        ref_paths.append(f"results/checkpoint-{i*10}")
    for i in range(1, 17):
        ref_paths.append(f"results2/checkpoint-{i*10}")
    for ref_path in ref_paths:
        ref_tokenizer = AutoTokenizer.from_pretrained("results2/final")
        ref_model = AutoModelForCausalLM.from_pretrained(ref_path).to(device)
        # Load data from valid.json
        with open('dataset/myvalid.json', 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        
        perplexities = []
        all_log_probs = []
        all_probs = []
        labels = []
        ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        mink_probs = dict()
        mink_log_probs = dict()
        mink_probs_base = dict()
        mink_log_probs_base = dict()
        mink_probs_ref = dict()
        mink_log_probs_ref = dict()
        for ratio in ratios:
            mink_probs[ratio] = []
            mink_log_probs[ratio] = []
            mink_probs_base[ratio] = []
            mink_log_probs_base[ratio] = []
            mink_probs_ref[ratio] = []
            mink_log_probs_ref[ratio] = []
        # Calculate perplexity
        for entry in tqdm(valid_data, desc="Calculating perplexities"):
            text = entry["text"]
            label = entry["label"]
            ppx, log_probs = calculatePerplexity(text, model, tokenizer, device)
            ref_ppx, ref_log_probs = calculatePerplexity(text, ref_model, ref_tokenizer, device)
            probs = np.exp(log_probs)
            ref_probs = np.exp(ref_log_probs)
            perplexities.append(ref_ppx - ppx)
            all_probs.append(np.mean(probs) - np.mean(ref_probs))
            all_log_probs.append(np.mean(log_probs) - np.mean(ref_log_probs))
            for ratio in ratios:
                mink_probs[ratio].append(calcmink(probs, ref_probs, ratio))
                mink_log_probs[ratio].append(calcmink(log_probs, ref_log_probs, ratio))
                mink_probs_base[ratio].append(calcmink_base(probs, ref_probs, ratio))
                mink_log_probs_base[ratio].append(calcmink_base(log_probs, ref_log_probs, ratio))
                mink_probs_ref[ratio].append(calcmink_ref(probs, ref_probs, ratio))
                mink_log_probs_ref[ratio].append(calcmink_ref(log_probs, ref_log_probs, ratio))
            labels.append(1 if label == "dirty" else 0)  # Convert labels to binary format
        
        # Calculate AUC for loss
        auc_ppl = roc_auc_score(labels, perplexities)
        auc_prob = roc_auc_score(labels, all_probs)
        auc_log_prob = roc_auc_score(labels, all_log_probs)
        print(f"ref_path: {ref_path}, PPL AUC: {auc_ppl}, PROB AUC: {auc_prob}, LOGPROB AUC: {auc_log_prob}")
        for ratio in ratios:
            auc_mink_prob = roc_auc_score(labels, mink_probs[ratio])
            auc_mink_log_prob = roc_auc_score(labels, mink_log_probs[ratio])
            print(f"ref_path: {ref_path}, RATIO: {ratio}, PROB AUC: {auc_mink_prob}, LOGPROB AUC: {auc_mink_log_prob}")
            auc_min_prob_base = roc_auc_score(labels, mink_probs_base[ratio])
            auc_min_log_prob_base = roc_auc_score(labels, mink_log_probs_base[ratio])
            print(f"ref_path: {ref_path}, RATIO: {ratio}, PROB_BASE AUC: {auc_min_prob_base}, LOGPROB_BASE AUC: {auc_min_log_prob_base}")
            auc_min_prob_ref = roc_auc_score(labels, mink_probs_ref[ratio])
            auc_min_log_prob_ref = roc_auc_score(labels, mink_log_probs_ref[ratio])
            print(f"ref_path: {ref_path}, RATIO: {ratio}, PROB_REF AUC: {auc_min_prob_ref}, LOGPROB_REF AUC: {auc_min_log_prob_ref}")
        print()