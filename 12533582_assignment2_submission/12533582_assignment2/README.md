# CSE5023 Assignment 2

Name: Weiqiang Duan

Student ID: 12533582

This directory contains the report, source code, datasets, checkpoints, and experiment outputs for Assignment 2.

All commands below should be executed from the assignment root directory after unzipping the submission:

```powershell
cd 12533582_assignment2
```

## Python Environment (Miniconda)

- OS: Windows
- Python=3.12
- pytorch=2.5.1. Install the torch and CUDA version supported by your computer.
- torchvision
- nltk=3.9.1
- numpy=1.26.3. Usually numpy is installed automatically with pytorch.
- matplotlib=3.10.0
- scikit-learn=1.8.0
- tqdm
- jieba

```powershell
python --version
```

To run the code, first activate your Python environment in a terminal, PowerShell, or cmd window. The commands below use `python` as the Python executable. Make sure `python` points to the intended Python 3.12 environment. If your Python executable is not on `PATH`, replace `python` in the commands with the absolute path to your local Python executable.

LaTeX is only needed if the report source is recompiled. The submitted PDF files are already included under `report/`.

## Directory Structure

```text
12533582_assignment2/
  code/
    part1_machine_translation/
    part2_sentiment_analysis/
    part3_vision_transformer/
  outputs/
    part1_machine_translation/
    part2_sentiment_analysis/
    part3_vision_transformer/
  report/
    main.tex
    main.pdf
    main_en.tex
    main_en.pdf
  code_notes/
```

## Part 1: English-to-Chinese Machine Translation

Main code:

- `code/part1_machine_translation/tokenization.py`
- `code/part1_machine_translation/model/transformer.py`
- `code/part1_machine_translation/translator_en2cn_baseline.py`
- `code/part1_machine_translation/translator_en2cn_cosine.py`
- `code/part1_machine_translation/translator_en2cn_ablation.py`
- `code/part1_machine_translation/part1_machine_translation.ipynb`
- `code/part1_machine_translation/chinese_bleu.ipynb`

Local dataset:

```text
code/part1_machine_translation/data/en-cn/
  train.txt
  dev.txt
  test.txt
```

Important outputs:

- Baseline: `outputs/part1_machine_translation/baseline_default_6_layer/`
- Cosine warm-up runs: `outputs/part1_machine_translation/cosine_default_6_layer_peak*/`
- Ablation runs: `outputs/part1_machine_translation/ablation/`
- Learnable PE run: `outputs/part1_machine_translation/pe_learnable_peak7e-4/`

Baseline command:

```powershell
python code\part1_machine_translation\translator_en2cn_baseline.py `
  --n-layer 6 --heads 8 --d-model 256 --d-ff 1024 `
  --epochs 20 --batch-size 64 --lr 0.0001 `
  --output-dir outputs\part1_machine_translation\baseline_default_6_layer
```

Cosine warm-up learning-rate tuning commands:

```powershell
$peaks = @("1e-4", "2e-4", "3e-4", "4e-4", "5e-4", "6e-4", "7e-4", "8e-4")
foreach ($peak in $peaks) {
  python code\part1_machine_translation\translator_en2cn_cosine.py `
    --name "cosine_default_6_layer_peak$peak" `
    --n-layer 6 --heads 8 --d-model 256 --d-ff 1024 `
    --epochs 20 --batch-size 64 --peak-lr $peak --warmup-epochs 5 --min-lr 1e-6 `
    --output-dir "outputs\part1_machine_translation\cosine_default_6_layer_peak$peak"
}
```

Ablation commands:

```powershell
python code\part1_machine_translation\translator_en2cn_ablation.py `
  --name ablation_default_peak7e-4_nlayer3 `
  --n-layer 3 --heads 8 --d-model 256 --d-ff 1024 `
  --epochs 20 --batch-size 64 --peak-lr 7e-4 --warmup-epochs 5 --min-lr 1e-6 `
  --output-dir outputs\part1_machine_translation\ablation\ablation_default_peak7e-4_nlayer3

python code\part1_machine_translation\translator_en2cn_ablation.py `
  --name ablation_default_peak7e-4_heads4 `
  --n-layer 6 --heads 4 --d-model 256 --d-ff 1024 `
  --epochs 20 --batch-size 64 --peak-lr 7e-4 --warmup-epochs 5 --min-lr 1e-6 `
  --output-dir outputs\part1_machine_translation\ablation\ablation_default_peak7e-4_heads4

python code\part1_machine_translation\translator_en2cn_ablation.py `
  --name ablation_default_peak7e-4_dmodel128 `
  --n-layer 6 --heads 8 --d-model 128 --d-ff 512 `
  --epochs 20 --batch-size 64 --peak-lr 7e-4 --warmup-epochs 5 --min-lr 1e-6 `
  --output-dir outputs\part1_machine_translation\ablation\ablation_default_peak7e-4_dmodel128
```

Learnable positional embedding command:

```powershell
python code\part1_machine_translation\translator_en2cn_cosine.py `
  --name pe_learnable_peak7e-4 `
  --pe-type learnable `
  --n-layer 6 --heads 8 --d-model 256 --d-ff 1024 `
  --epochs 20 --batch-size 64 --peak-lr 7e-4 --warmup-epochs 5 --min-lr 1e-6 `
  --output-dir outputs\part1_machine_translation\pe_learnable_peak7e-4
```

BLEU evaluation:

```text
Open code/part1_machine_translation/chinese_bleu.ipynb and run all cells.
The notebook reads the saved prediction files, applies Chinese tokenization, and records the BLEU results used in the report.
```

Integrated Part 1 notebook:

```text
Open code/part1_machine_translation/part1_machine_translation.ipynb.
It loads the dataset once, then runs the baseline, cosine warm-up, ablation, and learnable positional embedding experiments with separate output folders and checkpoint resume support.
Use USE_MINI_DATA=True for a quick check, or set USE_MINI_DATA=False for the full training files.
```

Best reported translation checkpoint for downstream use:

```text
outputs/part1_machine_translation/cosine_default_6_layer_peak7e-4/best_model.pt
```

## Part 2: Tweet Sentiment Analysis

Main code:

- `code/part2_sentiment_analysis/sentiment_model.py`
- `code/part2_sentiment_analysis/train_sentiment.py`

Local dataset:

```text
code/part2_sentiment_analysis/data/tweet_sentiment_extraction/
  train.jsonl
  test.jsonl
  README.md
```

Baseline command:

```powershell
python code\part2_sentiment_analysis\train_sentiment.py `
  --train-file code\part2_sentiment_analysis\data\tweet_sentiment_extraction\train.jsonl `
  --test-file code\part2_sentiment_analysis\data\tweet_sentiment_extraction\test.jsonl `
  --checkpoint outputs\part1_machine_translation\cosine_default_6_layer_peak7e-4\best_model.pt `
  --output-dir outputs\part2_sentiment_analysis\sentiment_baseline `
  --epochs 8 --batch-size 64 --lr 0.0002 --weight-decay 0.01
```

Without position embedding:

```powershell
python code\part2_sentiment_analysis\train_sentiment.py `
  --train-file code\part2_sentiment_analysis\data\tweet_sentiment_extraction\train.jsonl `
  --test-file code\part2_sentiment_analysis\data\tweet_sentiment_extraction\test.jsonl `
  --checkpoint outputs\part1_machine_translation\cosine_default_6_layer_peak7e-4\best_model.pt `
  --output-dir outputs\part2_sentiment_analysis\sentiment_without_pe `
  --epochs 8 --batch-size 64 --lr 0.0002 --weight-decay 0.01 `
  --no-position-embedding
```

Important outputs:

- `outputs/part2_sentiment_analysis/sentiment_baseline/`
- `outputs/part2_sentiment_analysis/sentiment_without_pe/`

## Part 3: Vision Transformer and CNN Comparison

Main code:

- `code/part3_vision_transformer/vit_model.py`
- `code/part3_vision_transformer/train_vit.py`
- `code/part3_vision_transformer/train_resnet.py`

CIFAR-10 is stored under:

```text
code/part3_vision_transformer/data/
```

ViT command:

```powershell
python code\part3_vision_transformer\train_vit.py `
  --epochs 20 --batch-size 128 `
  --output-dir outputs\part3_vision_transformer\vit_cifar10 `
  --amp --pin-memory --num-workers 0
```

Pretrained ResNet50 command:

```powershell
python code\part3_vision_transformer\train_resnet.py `
  --use-torch-hub --epochs 10 --batch-size 256 `
  --lr 0.001 --weight-decay 0.01 `
  --output-dir outputs\part3_vision_transformer\resnet50_cifar10 `
  --amp --pin-memory --num-workers 0
```

The ResNet50 script follows the assignment reference by loading:

```python
torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
```

Important outputs:

- `outputs/part3_vision_transformer/vit_cifar10/`
- `outputs/part3_vision_transformer/resnet50_cifar10/`

## Notes

- PowerShell may print profile execution policy warnings on startup; these can be ignored.
- The sentiment dataset is loaded from local JSONL files to avoid Windows access violation issues previously seen with `datasets.load_dataset`.
- The pretrained ResNet50 experiment may download PyTorch Hub model files if they are not already cached.
