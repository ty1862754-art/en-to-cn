import argparse
import json
import math
import random
import time
from pathlib import Path

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


def default_output_dir():
    return Path(__file__).resolve().parents[2] / "outputs" / "part1_machine_translation" / "baseline"


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
    """Corpus BLEU compatible with evaluate.load("bleu") for pre-tokenized strings."""
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0

    for prediction, reference_list in zip(predictions, references):
        pred_tokens = prediction.split()
        ref_tokens_list = [reference.split() for reference in reference_list]
        reference_length += min(len(ref_tokens) for ref_tokens in ref_tokens_list)
        translation_length += len(pred_tokens)

        merged_ref_ngram_counts = {}
        for ref_tokens in ref_tokens_list:
            ref_ngram_counts = get_ngrams(ref_tokens, max_order)
            for ngram, count in ref_ngram_counts.items():
                merged_ref_ngram_counts[ngram] = max(merged_ref_ngram_counts.get(ngram, 0), count)

        pred_ngram_counts = get_ngrams(pred_tokens, max_order)
        overlap = {
            ngram: min(count, merged_ref_ngram_counts.get(ngram, 0))
            for ngram, count in pred_ngram_counts.items()
        }
        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count
        for order in range(1, max_order + 1):
            possible_matches = len(pred_tokens) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    ratio = translation_length / reference_length if reference_length > 0 else 0.0
    bp = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / ratio) if ratio > 0.0 else 0.0
    bleu = geo_mean * bp
    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "length_ratio": ratio,
        "translation_length": translation_length,
        "reference_length": reference_length,
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


def evaluate_bleu(model, batches, data, max_len, device, prediction_path=None, max_examples=None, compute_bleu=True):
    model.eval()
    references = []
    predictions = []
    rows = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(batches, desc="Decoding", leave=False)):
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


def save_loss_curve(loss_history, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([row["epoch"] for row in loss_history], [row["train_loss"] for row in loss_history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Baseline Transformer Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the baseline English-to-Chinese Transformer.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--assets-dir", type=Path, default=default_assets_dir())
    parser.add_argument("--train-file", default="train.txt")
    parser.add_argument("--dev-file", default="dev.txt")
    parser.add_argument("--test-file", default="test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = args.data_dir / args.train_file
    dev_path = args.data_dir / args.dev_file
    test_path = args.data_dir / args.test_file

    print(f"Using device: {device}")
    print(f"Train file: {train_path}")
    print(f"Dev file: {dev_path}")
    print(f"Test file: {test_path}")

    data = PrepareData(str(train_path), str(dev_path), args.batch_size, UNK_ID, PAD_ID)
    test_data = PrepareData(str(train_path), str(test_path), args.batch_size, UNK_ID, PAD_ID)

    src_vocab_size = len(data.en_word_dict)
    tgt_vocab_size = len(data.cn_word_dict)

    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        args.seq_len,
        args.seq_len,
        args.d_model,
        args.n_layer,
        args.heads,
        args.dropout,
        args.d_ff,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-9)

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "train_size": len(data.train_en),
            "dev_size": len(data.dev_en),
            "test_size": len(test_data.dev_en),
        }
    )
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    config["assets_dir"] = str(args.assets_dir)

    with (args.output_dir / "baseline_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_bleu = -math.inf
    loss_history = []
    metrics_history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        iterator = tqdm(data.train_data, desc=f"Epoch {epoch:02d}/{args.epochs}")

        for batch in iterator:
            encoder_input = batch.src.to(device)
            decoder_input = batch.tgt.to(device)
            encoder_mask = batch.src_mask.to(device)
            decoder_mask = batch.tgt_mask.to(device)
            label = batch.tgt_y.to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ntokens = int(batch.ntokens.item())
            total_loss += loss.item() * ntokens
            total_tokens += ntokens
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_tokens, 1)
        epoch_record = {"epoch": epoch, "train_loss": train_loss}
        loss_history.append(epoch_record)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "en_word_dict": data.en_word_dict,
                    "cn_word_dict": data.cn_word_dict,
                    "en_index_dict": data.en_index_dict,
                    "cn_index_dict": data.cn_index_dict,
                    "epoch": epoch,
                    "train_loss": train_loss,
                },
                args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            dev_bleu, _, dev_bleu_result = evaluate_bleu(
                model,
                data.dev_data,
                data,
                args.seq_len,
                device,
                prediction_path=args.output_dir / "baseline_dev_predictions.txt",
                max_examples=args.eval_max_examples,
            )
            metric_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_bleu": dev_bleu,
                "dev_bleu_result": dev_bleu_result,
            }
            metrics_history.append(metric_record)
            print(f"Epoch {epoch}: dev_bleu={dev_bleu:.4f}")

            if dev_bleu > best_bleu:
                best_bleu = dev_bleu
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "en_word_dict": data.en_word_dict,
                        "cn_word_dict": data.cn_word_dict,
                        "en_index_dict": data.en_index_dict,
                        "cn_index_dict": data.cn_index_dict,
                        "best_epoch": epoch,
                        "best_dev_bleu": best_bleu,
                    },
                    args.output_dir / "baseline_best_model.pt",
                )

    with (args.output_dir / "baseline_loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "baseline_metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2)

    save_loss_curve(loss_history, args.assets_dir / "baseline_loss_curve.png")

    checkpoint = torch.load(args.output_dir / "baseline_best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_rows, _ = evaluate_bleu(
        model,
        test_data.dev_data,
        test_data,
        args.seq_len,
        device,
        prediction_path=args.output_dir / "baseline_test_predictions.txt",
        compute_bleu=False,
    )

    summary = {
        "best_epoch": checkpoint["best_epoch"],
        "best_dev_bleu": checkpoint["best_dev_bleu"],
        "final_train_loss": loss_history[-1]["train_loss"],
        "training_seconds": time.time() - start,
        "num_test_examples": len(test_rows),
        "sample_translations": test_rows[:5],
    }
    with (args.output_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
