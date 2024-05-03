from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TextDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
import torch
import numpy as np
import json
from tqdm import tqdm

# torch.cuda.set_device(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained("opt-350m")
model = AutoModelForCausalLM.from_pretrained("opt-350m", torch_dtype=torch.float32).to(device) # 将模型转移到GPU

# 假设你有一个包含文本行的列表 `text_lines`
with open("dataset/mytrain.json") as f:
    text_lines = json.load(f)
text_lines = [x["text"] for x in text_lines]

# 创建Dataset
dataset = Dataset.from_dict({"text": text_lines})


def group_texts(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(group_texts)
print(tokenized_dataset)

# 初始化DataCollator，这里使用MLM（masked language modeling）的方式作为示例，对于next token prediction，其实无需mask，但这个collator也适用
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",  # 输出目录
    overwrite_output_dir=False,
    num_train_epochs=3,  # 训练轮数
    per_device_train_batch_size=8,  # 每个GPU的训练批次大小
    save_steps=10,  # 每隔多少步保存一次模型
    save_total_limit=100,  # 保留的检查点的最大数量
    logging_dir="./logs",  # 日志目录
    evaluation_strategy="no",  # 由于是无监督任务，不需要评估
    weight_decay=0.01,  # 权重衰减
    learning_rate=5e-5,  # 学习率
    fp16=True,  # 是否使用混合精度训练
)

from transformers import Trainer, TrainingArguments

# 因为是语言建模任务，我们不需要定义任何模型的forward方法或计算损失，Trainer会自动处理
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 开始训练
trainer.train()


model.save_pretrained(f"results/final", safe_serialization=False)
tokenizer.save_pretrained(f"results/final", safe_serialization=False)