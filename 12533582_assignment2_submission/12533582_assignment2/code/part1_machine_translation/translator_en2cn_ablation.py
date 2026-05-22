import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import jieba
from tqdm import tqdm

from model.transformer import build_transformer
from tokenization import PrepareData, casual_mask


PAD_ID = 0
UNK_ID = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_data_dir():
    return Path(__file__).resolve().parent / "data" / "en-cn"


def default_output_base_dir():
    return Path(__file__).resolve().parents[2] / "outputs" / "part1_machine_translation" / "ablation"


def default_assets_dir():
    return Path(__file__).resolve().parents[2] / "assets"


def ids_to_tokens(ids, index_dict, skip_bos=True):
    tokens = []
    for idx in ids:
        token = index_dict.get(int(idx), "UNK")
        if skip_bos and token == "BOS":
            continue
        if token == "EOS":
            break
        if token != "PAD":
            tokens.append(token)
    return tokens


def segment_chinese(sentence):
    words = list(jieba.cut(sentence.strip(), cut_all=False))
    return " ".join(word for word in words if word.strip())


def get_ngrams(tokens, max_order):
    ngram_counts = {}
    for order in range(1, max_order + 1):
        for i in range(0, len(tokens) - order + 1):
            ngram = tuple(tokens[i : i + order])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts


def compute_bleu_like_evaluate(predictions, references, max_order=4, smooth=False):
    """Corpus BLEU compatible with evaluate.load("bleu") for pre-tokenized strings using sacrebleu."""
    import sacrebleu
    sacrebleu_refs = [[r[0] for r in references]]
    sacre_res = sacrebleu.corpus_bleu(predictions, sacrebleu_refs, tokenize="none")
    return {
        "bleu": sacre_res.score / 100.0,
        "precisions": [p / 100.0 for p in sacre_res.precisions],
        "brevity_penalty": sacre_res.bp,
        "length_ratio": sacre_res.sys_len / max(sacre_res.ref_len, 1),
        "translation_length": sacre_res.sys_len,
        "reference_length": sacre_res.ref_len,
    }


def greedy_decode(model, source, source_mask, tgt_word_dict, max_len, device):
    bos_id = tgt_word_dict["BOS"]
    eos_id = tgt_word_dict["EOS"]

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1, dtype=source.dtype, device=device).fill_(bos_id)

    while decoder_input.size(1) < max_len:
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1, dtype=source.dtype, device=device).fill_(next_word.item())],
            dim=1,
        )
        if next_word.item() == eos_id:
            break

    return decoder_input.squeeze(0).detach().cpu().tolist()


def evaluate_bleu(
    model,
    batches,
    data,
    max_len,
    device,
    prediction_path=None,
    max_examples=None,
    compute_bleu=True,
    show_progress=True,
):
    model.eval()
    references = []
    predictions = []
    rows = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(batches, desc="Decoding", leave=False, disable=not show_progress)):
            if max_examples is not None and i >= max_examples:
                break

            source = batch.src.to(device)
            source_mask = batch.src_mask.to(device)
            assert source.size(0) == 1, "BLEU evaluation expects batch_size=1."

            pred_ids = greedy_decode(model, source, source_mask, data.cn_word_dict, max_len, device)
            pred_tokens = ids_to_tokens(pred_ids, data.cn_index_dict)
            ref_tokens = ids_to_tokens(batch.tgt_y.squeeze(0).tolist(), data.cn_index_dict, skip_bos=False)
            src_tokens = ids_to_tokens(batch.src.squeeze(0).tolist(), data.en_index_dict)

            rows.append(
                {
                    "source": " ".join(src_tokens),
                    "reference": "".join(ref_tokens),
                    "prediction": "".join(pred_tokens),
                    "reference_tokens": " ".join(ref_tokens),
                    "prediction_tokens": " ".join(pred_tokens),
                }
            )
            if compute_bleu:
                predictions.append(segment_chinese(rows[-1]["prediction"]))
                references.append([segment_chinese(rows[-1]["reference"])])

    bleu_result = compute_bleu_like_evaluate(predictions, references) if compute_bleu and predictions else {"bleu": None}
    bleu = bleu_result["bleu"]

    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as f:
            f.write("source\treference\tprediction\treference_tokens\tprediction_tokens\n")
            for row in rows:
                f.write(
                    f"{row['source']}\t{row['reference']}\t{row['prediction']}\t"
                    f"{row['reference_tokens']}\t{row['prediction_tokens']}\n"
                )

    return bleu, rows, bleu_result


def get_warmup_cosine_lr(epoch, total_epochs, warmup_epochs, peak_lr, min_lr):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return peak_lr * epoch / warmup_epochs

    if total_epochs <= warmup_epochs:
        return min_lr

    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_loss_curve(loss_history, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([row["epoch"] for row in loss_history], [row["train_loss"] for row in loss_history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_lr_curve(lr_history, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([row["epoch"] for row in lr_history], [row["lr"] for row in lr_history], marker="o", color="#b45309")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Warm-up Cosine Learning Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



