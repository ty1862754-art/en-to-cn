# Part 2.2 Sentiment Position Embedding Results

## Goal

Compare sentiment classification performance with and without source-side positional encoding.

## Fixed Setup

Both runs should keep the same settings:

- Dataset: `code/part2_sentiment_analysis/data/tweet_sentiment_extraction/train.jsonl`
- Test set: `code/part2_sentiment_analysis/data/tweet_sentiment_extraction/test.jsonl`
- Checkpoint: `outputs/part1_machine_translation/cosine_default_6_layer_peak7e-4/best_model.pt`
- Epochs: 8
- Batch size: 64
- Learning rate: 0.0002
- Weight decay: 0.01
- Pooling: mean
- Seed: 42

Only the position embedding setting changes:

- `sentiment_baseline`: with sinusoidal position embedding.
- `sentiment_without_pe`: replaces position encoding with identity mapping.

## Result Table

Fill this table after `sentiment_without_pe` finishes.

| Experiment | Position embedding | Best dev acc | Test acc | Test macro F1 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `sentiment_baseline` | with PE | 0.7114 | 0.7097 | 0.7137 | 2 |
| `sentiment_without_pe` | without PE | TBD | TBD | TBD | TBD |

## Discussion Points

- If without PE is close to with PE, the tweet sentiment task may rely more on lexical cues than exact word order.
- If without PE is clearly worse, word order and local structures such as negation or contrast still matter.
- Compare this with Part 1 machine translation, where position information is more central because translation requires sequence order, alignment, and syntax-sensitive generation.
