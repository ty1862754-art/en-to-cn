import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from vit_model import VisionTransformer, count_parameters


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def default_data_dir():
    return Path(__file__).resolve().parents[2] / "code" / "part3_vision_transformer" / "data"


def default_output_dir():
    return Path(__file__).resolve().parents[2] / "outputs" / "part3_vision_transformer" / "vit_cifar10"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_train_valid_indices(targets, valid_size, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    valid_indices = []
    targets = np.asarray(targets)
    for class_id in sorted(np.unique(targets)):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        n_valid = int(round(len(class_indices) * valid_size))
        valid_indices.extend(class_indices[:n_valid].tolist())
        train_indices.extend(class_indices[n_valid:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    return train_indices, valid_indices


def build_transforms(augment):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_ops = []
    if augment:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_loaders(args):
    train_transform, eval_transform = build_transforms(augment=not args.no_augment)
    train_full = datasets.CIFAR10(root=args.data_dir, train=True, download=args.download, transform=train_transform)
    valid_full = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=args.download, transform=eval_transform)

    train_indices, valid_indices = stratified_train_valid_indices(train_full.targets, args.valid_size, args.seed)
    if args.max_train_samples and args.max_train_samples > 0:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_valid_samples and args.max_valid_samples > 0:
        valid_indices = valid_indices[: args.max_valid_samples]
    if args.max_test_samples and args.max_test_samples > 0:
        test_set = Subset(test_set, list(range(args.max_test_samples)))

    train_set = Subset(train_full, train_indices)
    valid_set = Subset(valid_full, valid_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, valid_loader, test_loader


def build_model(args):
    return VisionTransformer(
        image_size=32,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        pooling=args.pooling,
    )


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).sum().item(), labels.numel()


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, max_grad_norm, show_progress):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    iterator = tqdm(loader, desc="train", disable=not show_progress)
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if scaler is None:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_correct, batch_total = accuracy_from_logits(logits.detach(), labels)
        correct += batch_correct
        total += batch_total
        total_loss += loss.item() * labels.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0 for _ in CIFAR10_CLASSES]
    class_total = [0 for _ in CIFAR10_CLASSES]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            class_total[label] += 1
            class_correct[label] += int(label == pred)
    class_accuracy = {
        class_name: class_correct[idx] / class_total[idx] if class_total[idx] else 0.0
        for idx, class_name in enumerate(CIFAR10_CLASSES)
    }
    return total_loss / max(total, 1), correct / max(total, 1), class_accuracy


def save_curves(metrics, output_dir):
    epochs = [item["epoch"] for item in metrics]
    train_loss = [item["train_loss"] for item in metrics]
    valid_loss = [item["valid_loss"] for item in metrics]
    train_acc = [item["train_accuracy"] for item in metrics]
    valid_acc = [item["valid_accuracy"] for item in metrics]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, valid_loss, marker="o", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_acc, marker="o", label="train accuracy")
    plt.plot(epochs, valid_acc, marker="o", label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=180)
    plt.close()



