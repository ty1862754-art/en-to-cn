# Part 1.5 Position Embedding Experiment Notes

## Goal

Compare two positional encoding choices under the same machine translation training setup:

- `sinusoidal`: fixed sinusoidal positional encoding from the original Transformer.
- `learnable`: trainable positional embedding added to token embeddings.

## Fixed Training Setup

Use the default 6-layer structure selected after Part 1.4:

- `n_layer=6`
- `heads=8`
- `d_model=256`
- `d_ff=1024`
- `epochs=20`
- `batch_size=64`
- `peak_lr=7e-4`
- `warmup_epochs=5`
- `min_lr=1e-6`

## Expected Output Directories

- `outputs/part1_machine_translation/pe_sinusoidal_peak7e-4`
- `outputs/part1_machine_translation/pe_learnable_peak7e-4`

Each run writes:

- `best_model.pt`
- `checkpoint_epoch_*.pt`
- `config.json`
- `loss_history.json`
- `metrics_history.json`
- `lr_history.json`
- `bleu_result.json`
- `summary.json`
- `dev_predictions.txt`
- `test_predictions.txt`
- `loss_curve.png`
- `lr_curve.png`

## Result Table

Fill this table after both full runs finish.

| Experiment | PE type | final train loss | valid BLEU | test BLEU | best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `cosine_default_6_layer_peak7e-4` | sinusoidal | 0.2735 | 0.2486 | 0.2339 | 20 |
| `pe_learnable_peak7e-4` | learnable | 0.2559 | 0.2343 | 0.2229 | 20 |

## Report Analysis Points

- Sinusoidal PE provides a fixed positional prior and introduces no additional trainable parameters.
- Learnable PE adds trainable position parameters, which can adapt to the dataset but may need enough data and stable optimization.
- Compare validation BLEU and test BLEU together. If learnable PE improves training loss but not BLEU, it may be fitting the training distribution without better generalization.
- Keep all other settings unchanged so that the conclusion is about positional embedding rather than model capacity or learning rate.
