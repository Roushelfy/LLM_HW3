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
    ref_paths = ["results/checkpoint-40"]
    # ref_paths = ["gpt2-124m", "gpt2-355m"]
    # ref_paths = ["opt-125m", "opt-350m"]
    # for i in range(1, 10):
    #     ref_paths.append(f"results/checkpoint-{i*10}")

    for ref_path in ref_paths:
        ref_tokenizer = AutoTokenizer.from_pretrained("detected_model")
        # ref_tokenizer = AutoTokenizer.from_pretrained(ref_path)
        ref_model = AutoModelForCausalLM.from_pretrained(ref_path).to(device)
        # Load data from valid.json
        with open('dataset/myvalid.json', 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        temp = []
        for entry in valid_data:
            text = entry["text"]
            label = entry["label"]
            # if len(text) > 5000:
            #     print(len(text))
            temp.append({"text": text, "label": label})
        valid_data = temp
        clean_count = sum([entry["label"] == "clean" for entry in valid_data])

        perplexities = []
        all_log_probs = []
        all_probs = []
        labels = []
        ratios = [0.45, 0.46, 0.47, 0.48, 0.49,
                  0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
        mink_probs = dict()
        mink_log_probs = dict()
        mink_probs_base = dict()
        mink_log_probs_base = dict()
        mink_probs_ref = dict()
        mink_log_probs_ref = dict()
        min_score_dirty = dict()
        max_score_clean = dict()

        for ratio in ratios:
            mink_probs[ratio] = []
            mink_log_probs[ratio] = []
            mink_probs_base[ratio] = []
            mink_log_probs_base[ratio] = []
            mink_probs_ref[ratio] = []
            mink_log_probs_ref[ratio] = []
            min_score_dirty[ratio] = 1e10
            max_score_clean[ratio] = -1e10

        for entry in tqdm(valid_data, desc="Calculating perplexities"):
            text = entry["text"]
            label = entry["label"]
            # Convert labels to binary format
            labels.append(1 if label == "dirty" else 0)
            ppx, log_probs = calculatePerplexity(
                text, model, tokenizer, device)
            ref_ppx, ref_log_probs = calculatePerplexity(
                text, ref_model, ref_tokenizer, device)

            # ref_ppx, ref_log_probs = 0, [0 for x in log_probs]
            probs = np.exp(log_probs)
            ref_probs = np.exp(ref_log_probs)
            perplexities.append(ref_ppx - ppx)
            all_probs.append(np.mean(probs) - np.mean(ref_probs))
            all_log_probs.append(np.mean(log_probs) - np.mean(ref_log_probs))
            for ratio in ratios:
                mink_probs[ratio].append(calcmink(probs, ref_probs, ratio))
                mink_log_probs[ratio].append(
                    calcmink(log_probs, ref_log_probs, ratio))
                mink_probs_base[ratio].append(
                    calcmink_base(probs, ref_probs, ratio))
                mink_log_probs_base[ratio].append(
                    calcmink_base(log_probs, ref_log_probs, ratio))
                mink_probs_ref[ratio].append(
                    calcmink_ref(probs, ref_probs, ratio))
                mink_log_probs_ref[ratio].append(
                    calcmink_ref(log_probs, ref_log_probs, ratio))
                score = mink_log_probs_base[ratio][-1]
                if labels[-1] == 1:
                    min_score_dirty[ratio] = min(min_score_dirty[ratio], score)
                else:
                    max_score_clean[ratio] = max(max_score_clean[ratio], score)

        # Calculate AUC for loss
        auc_ppl = roc_auc_score(labels, perplexities)
        auc_prob = roc_auc_score(labels, all_probs)
        auc_log_prob = roc_auc_score(labels, all_log_probs)
        accuracy_ppl = accuracy_score(
            labels, generate_pred(perplexities, clean_count))
        accuracy_prob = accuracy_score(
            labels, generate_pred(all_probs, clean_count))
        accuracy_log_prob = accuracy_score(
            labels, generate_pred(all_log_probs, clean_count))
        print(f"ref_path: {ref_path}, PPL AUC: {auc_ppl}, PROB AUC: {auc_prob}, LOGPROB AUC: {auc_log_prob}, PPL ACC: {accuracy_ppl}, PROB ACC: {accuracy_prob}, LOGPROB ACC: {accuracy_log_prob}")
        for ratio in ratios:
            auc_mink_prob = roc_auc_score(labels, mink_probs[ratio])
            auc_mink_log_prob = roc_auc_score(labels, mink_log_probs[ratio])
            accuracy_mink_prob = accuracy_score(
                labels, generate_pred(mink_probs[ratio], clean_count))
            accuracy_mink_log_prob = accuracy_score(
                labels, generate_pred(mink_log_probs[ratio], clean_count))
            print(f"ref_path: {ref_path}, RATIO: {ratio}, PROB AUC: {auc_mink_prob}, LOGPROB AUC: {auc_mink_log_prob}, PROB ACC: {accuracy_mink_prob}, LOGPROB ACC: {accuracy_mink_log_prob}")
            auc_mink_prob_base = roc_auc_score(labels, mink_probs_base[ratio])
            auc_mink_log_prob_base = roc_auc_score(
                labels, mink_log_probs_base[ratio])
            accuracy_mink_prob_base = accuracy_score(
                labels, generate_pred(mink_probs_base[ratio], clean_count))
            accuracy_mink_log_prob_base = accuracy_score(
                labels, generate_pred(mink_log_probs_base[ratio], clean_count))
            print(f"ref_path: {ref_path}, RATIO: {ratio}, PROB_BASE AUC: {auc_mink_prob_base}, LOGPROB_BASE AUC: {auc_mink_log_prob_base}, PROB_BASE ACC: {accuracy_mink_prob_base}, LOGPROB_BASE ACC: {accuracy_mink_log_prob_base}")
            auc_mink_prob_ref = roc_auc_score(labels, mink_probs_ref[ratio])
            auc_mink_log_prob_ref = roc_auc_score(
                labels, mink_log_probs_ref[ratio])
            accuracy_mink_prob_ref = accuracy_score(
                labels, generate_pred(mink_probs_ref[ratio], clean_count))
            accuracy_mink_log_prob_ref = accuracy_score(
                labels, generate_pred(mink_log_probs_ref[ratio], clean_count))
            print(f"ref_path: {ref_path}, RATIO: {ratio}, PROB_REF AUC: {auc_mink_prob_ref}, LOGPROB_REF AUC: {auc_mink_log_prob_ref}, PROB_REF ACC: {accuracy_mink_prob_ref}, LOGPROB_REF ACC: {accuracy_mink_log_prob_ref}")
            print(f"score: {min_score_dirty[ratio] - max_score_clean[ratio]}")
            print()
        print("\n\n")
