import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sentiment_model import build_sentiment_model_from_checkpoint


UNK_ID = 0
PAD_ID = 1
BOS_TOKEN = "BOS"
EOS_TOKEN = "EOS"


def default_checkpoint_path():
    return (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "part1_machine_translation"
        / "cosine_default_6_layer_peak7e-4"
        / "best_model.pt"
    )


def default_output_dir():
    return Path(__file__).resolve().parents[2] / "outputs" / "part2_sentiment_analysis" / "sentiment_baseline"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_text(text):
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", str(text).lower())
    return [BOS_TOKEN] + tokens + [EOS_TOKEN]


class TweetSentimentDataset(Dataset):
    def __init__(self, rows, en_word_dict, label_to_id, seq_len, text_column, label_column):
        self.rows = rows
        self.en_word_dict = en_word_dict
        self.label_to_id = label_to_id
        self.seq_len = seq_len
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = tokenize_text(row[self.text_column])
        ids = [self.en_word_dict.get(token, UNK_ID) for token in tokens]
        ids = ids[: self.seq_len]
        attention = [1] * len(ids)
        if len(ids) < self.seq_len:
            pad_count = self.seq_len - len(ids)
            ids.extend([PAD_ID] * pad_count)
            attention.extend([0] * pad_count)

        label = self.label_to_id[str(row[self.label_column])]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.bool),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": str(row[self.text_column]),
            "raw_label": str(row[self.label_column]),
        }


def detect_columns(rows, text_column=None, label_column=None):
    if not rows:
        raise ValueError("No sentiment samples were loaded.")
    columns = set(rows[0].keys())
    if text_column is None:
        for candidate in ("text", "sentence", "tweet", "content"):
            if candidate in columns:
                text_column = candidate
                break
    if label_column is None:
        for candidate in ("label_text", "sentiment", "label", "labels"):
            if candidate in columns:
                label_column = candidate
                break
    if text_column not in columns or label_column not in columns:
        raise ValueError(f"Could not detect text/label columns from columns={sorted(columns)}.")
    return text_column, label_column


def load_sentiment_rows(args):
    if args.train_file:
        train_rows = read_table(args.train_file)
        dev_rows = read_table(args.dev_file) if args.dev_file else None
        test_rows = read_table(args.test_file) if args.test_file else None
        text_column, label_column = detect_columns(train_rows, args.text_column, args.label_column)
        if dev_rows is None or test_rows is None:
            train_rows, dev_rows, test_rows = split_rows(train_rows, label_column, args.seed)
        return train_rows, dev_rows, test_rows, text_column, label_column, "local_files"

    train_jsonl, test_jsonl = download_huggingface_jsonl(args.dataset_name)
    all_train_rows = read_jsonl(train_jsonl)
    test_rows = read_jsonl(test_jsonl)
    text_column, label_column = detect_columns(all_train_rows, args.text_column, args.label_column)
    labels = [str(row[label_column]) for row in all_train_rows]
    train_rows, dev_rows = train_test_split(
        all_train_rows,
        test_size=0.1,
        random_state=args.seed,
        stratify=labels,
    )
    return train_rows, dev_rows, test_rows, text_column, label_column, f"{args.dataset_name}:jsonl"


def download_huggingface_jsonl(dataset_name):
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass
    from huggingface_hub import hf_hub_download

    train_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename="train.jsonl")
    test_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename="test.jsonl")
    return Path(train_path), Path(test_path)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def split_rows(rows, label_column, seed):
    labels = [str(row[label_column]) for row in rows]
    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [str(row[label_column]) for row in temp_rows]
    dev_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_rows, dev_rows, test_rows


def maybe_limit(rows, max_samples):
    if max_samples is None or max_samples <= 0:
        return rows
    return rows[:max_samples]


def build_label_mapping(*splits, label_column):
    labels = sorted({str(row[label_column]) for rows in splits for row in rows})
    return {label: idx for idx, label in enumerate(labels)}


def train_one_epoch(model, loader, optimizer, loss_fn, device, max_grad_norm, show_progress):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    iterator = tqdm(loader, desc="train", disable=not show_progress)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader.dataset), 1), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    rows = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        pred_list = preds.cpu().tolist()
        label_list = labels.cpu().tolist()
        all_preds.extend(pred_list)
        all_labels.extend(label_list)
        for text, raw_label, pred_id, label_id in zip(batch["text"], batch["raw_label"], pred_list, label_list):
            rows.append({"text": text, "gold_label": raw_label, "gold_id": label_id, "pred_id": pred_id})

    loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds)
    return loss, acc, all_labels, all_preds, rows


def save_curves(metrics, output_dir):
    epochs = [item["epoch"] for item in metrics]
    train_loss = [item["train_loss"] for item in metrics]
    dev_loss = [item["dev_loss"] for item in metrics]
    train_acc = [item["train_accuracy"] for item in metrics]
    dev_acc = [item["dev_accuracy"] for item in metrics]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, dev_loss, marker="o", label="dev loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_acc, marker="o", label="train accuracy")
    plt.plot(epochs, dev_acc, marker="o", label="dev accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Part 1 Transformer encoder for tweet sentiment classification.")
    parser.add_argument("--dataset-name", default="mteb/tweet_sentiment_extraction")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--pooling", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-position-embedding", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_rows, dev_rows, test_rows, text_column, label_column, data_source = load_sentiment_rows(args)
    train_rows = maybe_limit(train_rows, args.max_train_samples)
    dev_rows = maybe_limit(dev_rows, args.max_dev_samples)
    test_rows = maybe_limit(test_rows, args.max_test_samples)
    label_to_id = build_label_mapping(train_rows, dev_rows, test_rows, label_column=label_column)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_sentiment_model_from_checkpoint(
        checkpoint,
        num_classes=len(label_to_id),
        seq_len=args.seq_len,
        dropout=args.dropout,
        pooling=args.pooling,
        use_position_embedding=not args.no_position_embedding,
    )
    if args.freeze_encoder:
        for module in (model.src_embed, model.src_pos, model.encoder):
            for param in module.parameters():
                param.requires_grad = False
    model.to(device)

    train_dataset = TweetSentimentDataset(train_rows, checkpoint["en_word_dict"], label_to_id, args.seq_len, text_column, label_column)
    dev_dataset = TweetSentimentDataset(dev_rows, checkpoint["en_word_dict"], label_to_id, args.seq_len, text_column, label_column)
    test_dataset = TweetSentimentDataset(test_rows, checkpoint["en_word_dict"], label_to_id, args.seq_len, text_column, label_column)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "data_source": data_source,
            "text_column": text_column,
            "label_column": label_column,
            "label_to_id": label_to_id,
            "train_size": len(train_dataset),
            "dev_size": len(dev_dataset),
            "test_size": len(test_dataset),
            "pretrained_checkpoint": str(args.checkpoint),
            "use_position_embedding": not args.no_position_embedding,
        }
    )
    for key in ("checkpoint", "output_dir", "train_file", "dev_file", "test_file"):
        if config[key] is not None:
            config[key] = str(config[key])
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_dev_acc = -1.0
    best_epoch = None
    metrics = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.max_grad_norm,
            show_progress=not args.no_progress,
        )
        dev_loss, dev_acc, _, _, _ = evaluate(model, dev_loader, loss_fn, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_acc,
        }
        metrics.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "best_epoch": best_epoch,
                    "best_dev_accuracy": best_dev_acc,
                },
                args.output_dir / "best_model.pt",
            )

    with (args.output_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_curves(metrics, args.output_dir)

    best = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_loss, test_acc, y_true, y_pred, test_rows_out = evaluate(model, test_loader, loss_fn, device)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        output_dict=True,
        zero_division=0,
    )
    for row in test_rows_out:
        row["pred_label"] = id_to_label[row["pred_id"]]
    errors = [row for row in test_rows_out if row["gold_id"] != row["pred_id"]][:30]

    summary = {
        "best_epoch": best_epoch,
        "best_dev_accuracy": best_dev_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report,
        "training_seconds": time.time() - start,
        "num_test_examples": len(test_dataset),
        "sample_errors": errors,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "test_errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
