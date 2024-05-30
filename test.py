from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score


def calculatePerplexity(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    # print(inputs, input_ids)
    with torch.no_grad():
        output = model(**inputs, labels=input_ids)
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
    k_length1 = max(int(len(vec)*ratio) - 1, 1)
    k_length2 = max(int(len(ref_vec)*ratio) - 1, 1)
    vec = np.array(vec)
    ref_vec = np.array(ref_vec)
    topk_vec = np.sort(vec)[:k_length1]
    topk_ref_vec = np.sort(ref_vec)[:k_length2]
    return np.mean(topk_vec) - np.mean(topk_ref_vec)


def calcmink_base(vec, ref_vec, ratio):
    # return 0
    k_length = max(int(len(vec)*ratio) - 1, 1)
    vec = np.array(vec)
    ref_vec = np.array(ref_vec)
    indexs = np.argpartition(vec, k_length)[:k_length]
    topk_vec = vec[indexs]
    topk_ref_vec = ref_vec[indexs]
    return np.mean(topk_vec) - np.mean(topk_ref_vec)


def calcmink_ref(vec, ref_vec, ratio):
    # return 0
    k_length = max(int(len(vec)*ratio) - 1, 1)
    vec = np.array(vec)
    ref_vec = np.array(ref_vec)
    indexs = np.argpartition(ref_vec, k_length)[:k_length]
    topk_vec = vec[indexs]
    topk_ref_vec = ref_vec[indexs]
    return np.mean(topk_vec) - np.mean(topk_ref_vec)


def generate_pred(vec, k):
    tmp = np.array(vec)
    indices = tmp.argsort()[:k]
    result = np.ones_like(tmp)
    result[indices] = 0
    return result


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("detected_model")
    model = AutoModelForCausalLM.from_pretrained("detected_model").to(device)

    ratio = 0.5

    ref_tokenizer = AutoTokenizer.from_pretrained("detected_model")
    ref_model = AutoModelForCausalLM.from_pretrained(
        "results/checkpoint-40").to(device)

    with open('dataset/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    for entry in tqdm(test_data, desc="Calculating perplexities"):
        text = entry["text"]
        ppx, log_probs = calculatePerplexity(
            text, model, tokenizer, device)
        ref_ppx, ref_log_probs = calculatePerplexity(
            text, ref_model, ref_tokenizer, device)

        # ref_ppx, ref_log_probs = 0, [0 for x in log_probs]
        probs = np.exp(log_probs)
        ref_probs = np.exp(ref_log_probs)
        mink_log_prob_base = calcmink_base(log_probs, ref_log_probs, ratio)
        entry["score"] = mink_log_prob_base

    with open('dataset/test_output.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
