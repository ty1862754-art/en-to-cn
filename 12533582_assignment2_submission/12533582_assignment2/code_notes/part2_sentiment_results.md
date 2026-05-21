# Part 2.1 Sentiment Fine-tuning Results

## Data

- Dataset: `mteb/tweet_sentiment_extraction`
- Source: `https://huggingface.co/datasets/mteb/tweet_sentiment_extraction`
- Default loader: `datasets.load_dataset("mteb/tweet_sentiment_extraction")`

## Encoder Checkpoint

The sentiment classifier uses the Part 1 machine translation encoder from:

`outputs/part1_machine_translation/cosine_default_6_layer_peak7e-4/best_model.pt`

This checkpoint is selected because it has the strongest Part 1 test BLEU among the stable sinusoidal-position-encoding default models.

## Model

- Source embedding + encoder loaded from the translation checkpoint.
- Decoder and projection layer are removed.
- Encoder output is pooled with mean pooling by default.
- A linear classification head maps the pooled vector to sentiment labels.

## Result Table

Fill this table after running the full experiment.

| Experiment | Pooling | Freeze encoder | Position Embedding | Best dev acc | Test acc | Best epoch |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `sentiment_baseline` | mean | no | default | 0.7114 | 0.7097 | 2 |
| `sentiment_without_pe` | mean | no | none | 0.7060 | 0.7148 | 3 |

## Output Directory

`outputs/part2_sentiment_analysis/sentiment_baseline`

Expected files:

- `best_model.pt`
- `config.json`
- `metrics_history.json`
- `summary.json`
- `test_errors.json`
- `loss_curve.png`
- `accuracy_curve.png`
