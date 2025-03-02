from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import pandas as pd


data_files = {
    "train": "../data/train.csv",
    "validation": "../data/val.csv",
    "test": "../data/test.csv",
}
dataset = load_dataset("csv", data_files=data_files)


tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

role_map = {
    0: "Regular User",
    1: "Cross-application Developer",
    2: "Inner-application Developer",
}


def preprocess_function(examples):

    id = ["" if text is None else text for text in examples["creator"]]
    role_texts = [role_map[r] if r in role_map else "" for r in examples["role"]]
    freq = ["" if text is None else text for text in examples["creator_freq"]]
    profile = [f"{i} {r} {f}" for i, r, f in zip(id, role_texts, freq)]


    summaries = ["" if text is None else text for text in examples["summary"]]
    descriptions = ["" if text is None else text for text in examples["description"]]

    return tokenizer(
        profile, summaries, descriptions,
        padding="max_length",
        truncation=True,
        max_length=512
    )



tokenized_datasets = dataset.map(preprocess_function, batched=True)


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


tokenized_datasets = tokenized_datasets.map(encode_labels)
tokenized_datasets = tokenized_datasets.remove_columns(["id", "summary", "description", "resolution", "product", "role"])
tokenized_datasets.set_format("torch")


train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]


model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary", pos_label=0),
        "recall": recall_score(labels, predictions, average="binary", pos_label=0),
        "f1": f1_score(labels, predictions, average="binary", pos_label=0),
    }


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.05,
    save_total_limit=2,
    load_best_model_at_end=True,
)

model.config.hidden_dropout_prob = 0.3
model.config.attention_probs_dropout_prob = 0.3


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


predictions, labels, metrics = trainer.predict(test_dataset)


predicted_labels = torch.argmax(torch.tensor(predictions), dim=1).numpy()


accuracy = accuracy_score(labels, predicted_labels)
precision_positive = precision_score(labels, predicted_labels, pos_label=0)
precision_negative = precision_score(labels, predicted_labels, pos_label=1)
recall_positive = recall_score(labels, predicted_labels, pos_label=0)
recall_negative = recall_score(labels, predicted_labels, pos_label=1)
f1_positive = f1_score(labels, predicted_labels, pos_label=0)
f1_negative = f1_score(labels, predicted_labels, pos_label=1)




print(f"Accuracy: {accuracy:.4f}")
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
    "Metric": ["Accuracy", "Positive Precision", "Positive Recall", "Positive F1", "Negative Precision", "Negative Recall", "Negative F1"],
    "Value": [accuracy, precision_positive, recall_positive, f1_positive, precision_negative, recall_negative, f1_negative],
})
metrics_path = "metrics.csv"
metrics_df.to_csv(metrics_path, index=False)