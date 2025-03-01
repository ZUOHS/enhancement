from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import pandas as pd

data_files = {
    "train": "../data/train.csv",
    "validation": "../data/val.csv",
    "test": "../data/test.csv",
}
dataset = load_dataset("csv", data_files=data_files)

# 加载 XLNet 分词器
tokenizer = ElectraTokenizer.from_pretrained("electra-base-discriminator")

role_map = {
    0: "Regular User",
    1: "Cross-application Developer",
    2: "Inner-application Developer",
}


def preprocess_function(examples):
    # 先检查 role 是否为空，避免 role_map[r] 时报错
    id = ["" if text is None else text for text in examples["creator"]]
    role_texts = [role_map[r] if r in role_map else "" for r in examples["role"]]
    freq = ["" if text is None else text for text in examples["creator_freq"]]
    profile = [f"{i} {r} {f}" for i, r, f in zip(id, role_texts, freq)]

    # 处理可能为空的 summary 和 description
    summaries = ["" if text is None else text for text in examples["summary"]]
    descriptions = ["" if text is None else text for text in examples["description"]]

    return tokenizer(
        profile, summaries, descriptions,
        padding="max_length",
        truncation=True,
        max_length=512
    )


# 对数据集进行分词处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 将标签映射为整数
label_map = {
    "INVALID": 1,
    "DUPLICATE": 1,
    "FIXED": 0,
    "WONTFIX": 1,
    "INCOMPLETE": 1,
    "WORKSFORME": 1,
    "EXPIRED": 1,
    "MOVED": 1,
    "INACTIVE": 1,
}
def encode_labels(example):
    example["label"] = label_map[example["resolution"]]
    return example

# 映射标签并移除不需要的列
tokenized_datasets = tokenized_datasets.map(encode_labels)
tokenized_datasets = tokenized_datasets.remove_columns(["id", "summary", "description", "resolution", "product", "role"])
tokenized_datasets.set_format("torch")

# 数据集分割
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

# 加载 XLNet 模型
model = ElectraForSequenceClassification.from_pretrained("electra-base-discriminator", num_labels=2)

# 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary", pos_label=0),
        "recall": recall_score(labels, predictions, average="binary", pos_label=0),
        "f1": f1_score(labels, predictions, average="binary", pos_label=0),
    }

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)


# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 训练完成后保存模型


print(f"训练 XLNet 模型结束")

# 在测试集上评估模型，并保存结果到 CSV 文件
print("正在评估测试集并保存结果到 CSV 文件...")
predictions, labels, metrics = trainer.predict(test_dataset)

# 转换预测结果为类别
predicted_labels = torch.argmax(torch.tensor(predictions), dim=1).numpy()

# 计算详细指标
accuracy = accuracy_score(labels, predicted_labels)
precision_positive = precision_score(labels, predicted_labels, pos_label=0)
precision_negative = precision_score(labels, predicted_labels, pos_label=1)
recall_positive = recall_score(labels, predicted_labels, pos_label=0)
recall_negative = recall_score(labels, predicted_labels, pos_label=1)
f1_positive = f1_score(labels, predicted_labels, pos_label=0)
f1_negative = f1_score(labels, predicted_labels, pos_label=1)

try:
    auc = roc_auc_score(labels, predictions[:, 1])
except ValueError:
    auc = None


print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}" if auc else "AUC: error")
print(f"Positive Precision: {precision_positive:.4f}, Recall: {recall_positive:.4f}, F1: {f1_positive:.4f}")
print(f"Negative Precision: {precision_negative:.4f}, Recall: {recall_negative:.4f}, F1: {f1_negative:.4f}")


test_results_df = pd.DataFrame({
    "id": dataset["test"]["id"],                  
    "summary": dataset["test"]["summary"],      
    "description": dataset["test"]["description"],  
    "true_label": labels,                         
    "predicted_label": predicted_labels,         
})


output_path = "results.csv"
test_results_df.to_csv(output_path, index=False)


metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Positive Precision", "Positive Recall", "Positive F1", "Negative Precision", "Negative Recall", "Negative F1"],
    "Value": [accuracy, auc, precision_positive, recall_positive, f1_positive, precision_negative, recall_negative, f1_negative],
})
metrics_path = "metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
