from pathlib import Path
import sys

import torch
import torch.nn as nn


PART1_DIR = Path(__file__).resolve().parents[1] / "part1_machine_translation"
if str(PART1_DIR) not in sys.path:
    sys.path.insert(0, str(PART1_DIR))

from model.transformer import build_transformer  # noqa: E402


class IdentityPosition(nn.Module):
    def forward(self, x):
        return x


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        num_classes,
        d_model=256,
        n_layer=6,
        heads=8,
        dropout=0.1,
        d_ff=1024,
        pe_type="sinusoidal",
        use_position_embedding=True,
        pooling="mean",
    ):
        super().__init__()
        self.pooling = pooling
        base = build_transformer(
            src_vocab_size,
            tgt_vocab_size,
            src_seq_len,
            tgt_seq_len,
            d_model,
            n_layer,
            heads,
            dropout,
            d_ff,
            pe_type,
        )
        self.src_embed = base.src_embed
        self.src_pos = base.src_pos if use_position_embedding else IdentityPosition()
        self.encoder = base.encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        src_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        x = self.src_embed(input_ids)
        x = self.src_pos(x)
        encoded = self.encoder(x, src_mask)
        pooled = self.pool(encoded, attention_mask)
        return self.classifier(pooled)

    def pool(self, encoded, attention_mask):
        if self.pooling == "first":
            return encoded[:, 0]
        if self.pooling == "last":
            lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_index = torch.arange(encoded.size(0), device=encoded.device)
            return encoded[batch_index, lengths]
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        mask = attention_mask.unsqueeze(-1).type_as(encoded)
        summed = (encoded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


def build_sentiment_model_from_checkpoint(
    checkpoint,
    num_classes,
    seq_len=60,
    dropout=0.1,
    pooling="mean",
    use_position_embedding=True,
):
    config = checkpoint.get("config", {})
    src_vocab_size = len(checkpoint["en_word_dict"])
    tgt_vocab_size = len(checkpoint["cn_word_dict"])

    model = TransformerEncoderClassifier(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=config.get("seq_len", seq_len),
        tgt_seq_len=config.get("seq_len", seq_len),
        num_classes=num_classes,
        d_model=config.get("d_model", 256),
        n_layer=config.get("n_layer", 6),
        heads=config.get("heads", 8),
        dropout=dropout,
        d_ff=config.get("d_ff", 1024),
        pe_type=config.get("pe_type", "sinusoidal"),
        use_position_embedding=use_position_embedding,
        pooling=pooling,
    )

    state = checkpoint["model_state_dict"]
    model.src_embed.load_state_dict(_strip_prefix(state, "src_embed."))
    if use_position_embedding:
        model.src_pos.load_state_dict(_strip_prefix(state, "src_pos."), strict=False)
    model.encoder.load_state_dict(_strip_prefix(state, "encoder."))
    return model


def _strip_prefix(state_dict, prefix):
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
