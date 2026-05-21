# CSE5023 Assignment 2 作业完成计划

来源文件：`E:\code\DPfinal2PG\cse5023_final.pdf`

本计划用于指导 `12533582_assignment2` 的代码整理、实验执行和报告撰写。重点不是只列题目要求，而是把每一题的输入、处理步骤、输出产物，以及它如何成为下一题的输入写清楚。

## 0. 总体提交目标

本次作业分为三部分：

1. 机器翻译：实现并训练 English-to-Chinese Transformer。
2. 情感分析：复用第一部分训练好的 Transformer encoder 做分类微调。
3. Vision Transformer：在 CIFAR10 上实现 ViT，并与 CNN/ResNet50 对比。

最终提交目录建议保持如下结构：

```text
12533582_assignment2/
  code/
    part1_machine_translation/
    part2_sentiment_analysis/
    part3_vision_transformer/
  report/
    main.tex
    main.pdf
  assets/
  code_notes/
  README.md
  assignment_completion_plan.md
```

关键原则：

1. 前一题产生的模型、词表、配置和实验结果，要明确保存路径，因为后一题可能直接加载。
2. 每个实验都要同时保存代码、配置、模型权重、指标结果和报告可引用的结论。
3. 报告中出现的数值必须能在 `assets/`、`code_notes/` 或实验输出目录中找到来源。

## 1. 第一部分：机器翻译（50 分）

### 1.1 Transformer 架构实现

本题是后续所有机器翻译实验的代码基础，也是第二部分情感分析中 encoder 预训练参数的来源。

输入：

- 当前提交目录中的待补全代码：`code/part1_machine_translation/tokenization.py`
- 当前提交目录中的待补全代码：`code/part1_machine_translation/model/transformer.py`
- 当前提交目录中的训练/验证/测试语料：`code/part1_machine_translation/data/en-cn/`

操作流程：

1. 确认代码和数据都已经整理到 `code/part1_machine_translation/`，后续实验不再依赖外部原始文件夹。
2. 在 `tokenization.py` 中补全 `wordToID`，把英文和中文 token 序列转换为 id 序列。
3. 确认 `MaskBatch` 能生成三类训练所需张量：`src`、`tgt`、`tgt_y`。
4. 确认 encoder padding mask 的形状为 `[B, 1, 1, src_len]`。
5. 确认 decoder mask 同时包含 padding mask 和 causal mask，形状为 `[B, 1, tgt_len, tgt_len]`。
6. 在 `model/transformer.py` 中补全 `FeedForwardBlock`。
7. 在 `MultiHeadAttentionBlock.attention` 中补全 scaled dot-product attention、mask、softmax 和 value 加权求和。
8. 用小 batch 做一次 forward smoke test，确认 `encode -> decode -> project` 能跑通。

输出：

- `code/part1_machine_translation/tokenization.py`
- `code/part1_machine_translation/model/transformer.py`
- 1.1 报告文字：说明修改顺序、Encoder/Decoder 差异、mask 实现、训练和推理阶段差异。

输出给下一题：

- 1.2 直接使用 1.1 完成的 `tokenization.py` 和 `model/transformer.py` 训练 baseline 翻译模型。

### 1.2 模型训练与 BLEU 评估

本题产生机器翻译 baseline 模型。该模型是 1.3、1.4、1.5 的比较基准，也为 2.1 提供可加载的 encoder 参数。

输入：

- 1.1 完成的 Transformer 架构代码。
- 数据集：`code/part1_machine_translation/data/en-cn/train.txt`、`dev.txt`、`test.txt`。
- 训练脚本：建议整理为 `code/part1_machine_translation/translator_en2cn.py`。
- BLEU 评估脚本：`chinese_bleu.ipynb` 或等价 `.py` 脚本。

操作流程：

1. 固定 baseline 配置，例如 `n_layer`、`h_num`、`d_model`、`d_ff`、`dropout`、`batch_size`、`lr`、`epoch`。
2. 运行训练脚本，从零训练 English-to-Chinese Transformer。
3. 每个 epoch 记录 train loss。
4. 按固定间隔保存中间模型 checkpoint，例如每 5 个 epoch 保存一次 `checkpoint_epoch_005.pt`、`checkpoint_epoch_010.pt`，用于满足作业中“合适轮次保存中间模型”的要求，也方便训练中断后恢复或比较不同阶段翻译效果。
5. 按固定间隔在验证集上做 greedy decoding，并计算 validation BLEU。建议与 checkpoint 保存间隔保持一致，例如每 5 个 epoch 验证一次。
6. 保存最优模型权重 `baseline_best_model.pt`，保存依据为 validation BLEU。
7. 用最优模型在 test set 上生成预测文本。
8. 对中文参考译文和模型输出先分词，再计算 test BLEU。
9. 选取若干测试样例，记录英文输入、参考中文、模型输出，并分析翻译好坏。

输出：

- `outputs/part1_machine_translation/baseline/baseline_best_model.pt`
- `outputs/part1_machine_translation/baseline/checkpoint_epoch_*.pt`
- `outputs/part1_machine_translation/baseline/baseline_config.json`
- `outputs/part1_machine_translation/baseline/baseline_loss_history.json`
- `outputs/part1_machine_translation/baseline/baseline_metrics_history.json`
- `outputs/part1_machine_translation/baseline/baseline_test_predictions.txt`
- `assets/baseline_loss_curve.png`（如果报告需要图片）
- `code_notes/part1_baseline_results.md`

输出给下一题：

- 1.3 使用 baseline 配置和 baseline BLEU 作为学习率调度实验的对照组。
- 1.4 使用 baseline 配置作为消融实验的固定参考。
- 1.5 使用 baseline 中的 sinusoidal position encoding 结果作为位置嵌入对照。
- 2.1 加载 `baseline_best_model.pt` 中的 encoder 参数，作为情感分析预训练 encoder。

### 1.3 预热策略与学习率调优

本题是在 1.2 baseline 训练流程上只改变学习率策略，用 BLEU 判断 warm-up cosine 是否有效。

输入：

- 1.2 的 baseline 代码、baseline 配置和 baseline 指标。
- 1.2 输出的 test BLEU，作为对照。

操作流程：

1. 在训练脚本中加入 cosine annealing with warm-up scheduler。
2. Warm-up 阶段让学习率线性上升到 peak learning rate。
3. Warm-up 之后按 cosine 曲线下降到最小学习率。
4. 至少设置两组不同调度参数，例如不同 `peak_lr`、`warmup_epoch` 或 `warmup_steps`。
5. 每组实验保持模型结构、数据划分、batch size 和 epoch 尽量一致，只改变学习率策略。
6. 保存每组实验的 loss、validation BLEU、test BLEU。
7. 与 1.2 baseline 对比，选择 BLEU 最好的模型作为后续可选增强基准。

输出：

- `save/models/cosine_*/best_model.pt`
- `save/models/cosine_*/config.json`
- `save/models/cosine_*/loss_history.json`
- `code_notes/part1_cosine_results.md`

输出给下一题：

- 1.4 消融实验可以继续使用 1.2 baseline，也可以在报告中说明采用 1.3 中 BLEU 更高的配置作为增强 baseline。
- 1.5 位置嵌入实验应优先使用同一套学习率策略，避免把学习率差异误认为位置嵌入差异。

### 1.4 超参数消融实验

本题分析模型结构参数对翻译效果的影响。它依赖 1.2 或 1.3 中已经确定的 baseline 配置。

输入：

- 1.2 baseline 配置，或 1.3 中选出的最佳学习率策略。
- 固定的数据集划分和评估脚本。

操作流程：

1. 先确定一组 baseline 配置，并记录完整参数。
2. 每次只改变一个变量，其他变量保持不变。
3. 至少完成以下三类实验：
   - 改变 Transformer block 数量，例如 `n_layer=3` vs baseline。
   - 改变 attention head 数量，例如 `h_num=16` vs baseline。
   - 改变 embedding dimension，例如 `d_model=128` 或更大维度 vs baseline。
4. 每组实验训练完成后，用同一套 BLEU 脚本评估 validation/test BLEU。
5. 对比 train loss 和 BLEU，判断模型是否欠拟合、过拟合或泛化变差。

输出：

- `save/models/ablation_*/best_model.pt`
- `save/models/ablation_*/config.json`
- `save/models/ablation_*/metrics.json`
- `code_notes/part1_ablation_results.md`

输出给下一题：

- 1.5 位置嵌入实验应使用 1.4 中确定的固定结构参数，避免结构变化影响位置嵌入结论。
- 报告中 1.4 的结论可帮助解释 1.5 中位置嵌入变化是否受模型容量影响。

### 1.5 位置嵌入实验

本题比较不同位置嵌入方式对机器翻译的影响。它应基于前面已经稳定的训练流程和固定模型结构。

输入：

- 1.1 中的 `PositionalEncoding` 实现。
- 1.2/1.3/1.4 中确定的固定训练配置。
- baseline sinusoidal position encoding 的 loss 和 BLEU。

操作流程：

1. 在 `model/transformer.py` 中新增 learnable positional embedding。
2. 在 `build_transformer` 中增加参数，例如 `pe_type='sinusoidal'` 或 `pe_type='learnable'`。
3. 保持其他训练参数不变，分别训练 sinusoidal PE 和 learnable PE。
4. 分别记录 train loss、validation BLEU、test BLEU。
5. 分析两种位置嵌入的差异：固定先验、可学习参数、泛化能力、数据规模影响。

输出：

- `save/models/pe_sinusoidal/best_model.pt`
- `save/models/pe_learnable/best_model.pt`
- `save/models/pe_*/metrics.json`
- `code_notes/part1_position_embedding_results.md`

输出给下一题：

- 如果 2.1 使用第一部分 encoder 参数，需要明确选择哪个模型作为 encoder 来源：baseline、cosine 最优、ablation 最优，或 learnable PE 版本。
- 建议优先选择 validation/test BLEU 最好的翻译模型，并在 2.1 报告中说明 encoder checkpoint 来源。

## 2. 第二部分：情感分析（30 分）

第二部分不是独立从零开始。它的核心输入是第一部分训练好的 Transformer encoder。

### 2.1 模型微调

输入：

- 第一部分选定的翻译模型 checkpoint，例如 `save/models/.../best_model.pt`。
- 第一部分的 encoder 结构代码和词表/tokenization 逻辑。
- 情感分析数据集：Tweet Sentiment Extraction，来源为 Hugging Face 数据集
  `mteb/tweet_sentiment_extraction`：
  `https://huggingface.co/datasets/mteb/tweet_sentiment_extraction`。
  该数据集提供 tweet 文本与情感标签，可用于构建 train/validation/test
  情感分类样本。实现时优先使用 `datasets.load_dataset("mteb/tweet_sentiment_extraction")`
  下载；若网络不可用，则先手动下载或缓存到本地，再通过脚本参数指定本地数据路径。

操作流程：

1. 新建 `code/part2_sentiment_analysis/`。
2. 复用或改造第一部分的 encoder 代码，去掉 decoder 和 projection layer。
3. 加载第一部分 checkpoint 中 encoder 相关参数。
4. 根据情感分析数据构建文本 tokenization、padding 和 attention mask。
5. 在 encoder 输出后加入 pooling 方式，例如 mean pooling、first token pooling 或最后有效 token pooling。
6. 添加线性分类层，将 hidden state 映射到情感类别数。
7. 使用 `CrossEntropyLoss` 训练分类模型。
8. 每个 epoch 记录 train loss 和 validation accuracy。
9. 在 test set 上计算最终 accuracy，并保存若干错误案例。

输出：

- `code/part2_sentiment_analysis/sentiment_model.py`
- `code/part2_sentiment_analysis/train_sentiment.py`
- `save/sentiment/best_model.pt`
- `save/sentiment/metrics.json`
- `code_notes/part2_sentiment_results.md`

输出给下一题：

- 2.2 在同一套 sentiment 训练代码上改变是否使用 position embedding，因此 2.1 的训练流程和评估脚本要保持可复用。

### 2.2 位置嵌入探究

输入：

- 2.1 的情感分类训练脚本。
- 2.1 中确定的 encoder + classifier 结构。

操作流程：

1. 设置 with position embedding 版本，保留 encoder 中的位置嵌入。
2. 设置 without position embedding 版本，移除或置零位置嵌入。
3. 两组实验保持相同随机种子、batch size、learning rate、epoch 和数据划分。
4. 分别记录 train loss、validation accuracy、test accuracy。
5. 分析情感分类任务是否依赖词序，以及该现象与机器翻译任务中的位置嵌入作用有何不同。

输出：

- `save/sentiment_with_pe/metrics.json`
- `save/sentiment_without_pe/metrics.json`
- `code_notes/part2_position_embedding_results.md`

### 2.3 可选进阶优化

输入：

- 2.1 或 2.2 中表现较好的情感分析模型。

可选操作：

- Early stopping
- Gradient clipping
- Dropout 调优
- Label smoothing
- 冻结 encoder 前几层，只训练分类头
- 先训练分类头，再整体 fine-tune

输出：

- 一组明确优于 2.1 baseline 或能解释现象的可选实验。
- `code_notes/part2_optional_results.md`

## 3. 第三部分：Vision Transformer（20 分）

第三部分不依赖第一、第二部分的权重，但依赖 Transformer encoder 的思想。代码应单独放在 `code/part3_vision_transformer/`。

### 3.1 ViT 实现

输入：

- CIFAR10 数据集。数据集与训练流程参考 PyTorch 官方 CIFAR10 tutorial：
  `https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`。
  实现时优先使用 `torchvision.datasets.CIFAR10` 自动下载并构建
  train/valid/test DataLoader。
- Transformer encoder 结构思想。

操作流程：

1. 构建 CIFAR10 train/valid/test DataLoader。
2. 将图片切分为 patch，例如 32x32 图像切成 4x4 或 8x8 patch。
3. 将每个 patch flatten 后输入线性层，得到 patch embedding。
4. 加入 learnable position embedding。
5. 可选加入 class token，用 class token 输出做分类；也可对所有 patch 输出做 mean pooling。
6. 将 patch 序列输入 Transformer encoder。
7. 接分类头输出 10 类 logits。
8. 使用 CrossEntropyLoss 训练，记录 train loss 和 validation accuracy。
9. 在 test set 上报告最终 accuracy。

输出：

- `code/part3_vision_transformer/vit_model.py`
- `code/part3_vision_transformer/train_vit.py`
- `save/vit/best_model.pt`
- `save/vit/metrics.json`
- `code_notes/part3_vit_results.md`

输出给下一题：

- 3.2 使用 3.1 的 ViT 指标作为与 ResNet50/CNN 对比的基准。

### 3.2 与 CNN/ResNet50 对比

输入：

- 3.1 的 ViT test accuracy、参数量、训练耗时。
- CIFAR10 相同数据划分。
- 预训练 ResNet50 或其他 CNN baseline。CIFAR10 数据读取和基础训练流程继续参考
  PyTorch 官方 CIFAR10 tutorial：
  `https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`。

操作流程：

1. 加载 torchvision 中的 ResNet50 预训练模型。
2. 修改最后一层分类头，使输出类别数为 10。
3. 使用与 ViT 相同的数据划分和尽量相同的训练配置。
4. 训练或 fine-tune ResNet50。
5. 记录 train loss、validation accuracy、test accuracy、参数量和训练耗时。
6. 与 3.1 的 ViT 结果放在同一张表中比较。
7. 分析 ViT 和 CNN 在 CIFAR10 小图像数据上的优势与不足。

输出：

- `code/part3_vision_transformer/train_resnet.py`
- `save/resnet50/best_model.pt`
- `save/resnet50/metrics.json`
- `code_notes/part3_resnet_comparison.md`

## 4. 推荐执行顺序

1. 完成 1.1：补全 Transformer 架构代码，并写清楚结构与 mask。
2. 完成 1.2：训练 baseline，保存 checkpoint、loss、BLEU 和样例翻译。
3. 完成 1.3：在 baseline 基础上加入 warm-up cosine，选出较优学习率策略。
4. 完成 1.4：固定 baseline 或最佳 scheduler，做结构消融。
5. 完成 1.5：固定结构和训练流程，只比较位置嵌入。
6. 从 1.2-1.5 中选择一个最合理的 encoder checkpoint，作为 2.1 输入。
7. 完成 2.1：加载 encoder，添加分类头，完成情感分类 baseline。
8. 完成 2.2：在情感分类中比较有无位置嵌入。
9. 视时间完成 2.3：做一个小而明确的优化。
10. 完成 3.1：实现并训练 ViT。
11. 完成 3.2：训练 ResNet50/CNN，对比 ViT。
12. 汇总所有实验结果到 `code_notes/`，将报告引用的图表或结果放入 `assets/`。
13. 最后检查 `README.md`，确保每一部分都有运行命令、输入路径和输出路径。

## 5. 报告章节建议

1. 引言：说明三个任务的关系。
2. 第一部分：English-to-Chinese 机器翻译
   - 1.1 Transformer 架构实现
   - 1.2 训练流程与 BLEU 评估
   - 1.3 Warm-up Cosine 学习率调优
   - 1.4 超参数消融实验
   - 1.5 位置嵌入实验
3. 第二部分：情感分析
   - 2.1 Encoder 微调模型
   - 2.2 位置嵌入影响
   - 2.3 可选进阶技术
4. 第三部分：Vision Transformer
   - 3.1 ViT 实现
   - 3.2 ViT 与 ResNet50 对比
5. 总结与反思：说明任务间如何复用 Transformer，以及实验中观察到的主要现象。
6. 代码运行说明：列出每个脚本的运行命令、输入和输出。

## 6. 每题产物检查清单

| 题目 | 必须有的代码 | 必须有的实验产物 | 必须写入报告 |
| --- | --- | --- | --- |
| 1.1 | `tokenization.py`, `model/transformer.py` | forward smoke test 结果 | 架构、mask、训练/推理差异 |
| 1.2 | baseline 训练和 BLEU 脚本 | baseline checkpoint、loss、BLEU、翻译样例 | loss、BLEU、案例分析 |
| 1.3 | scheduler 训练脚本 | 多组 scheduler 指标 | 学习率策略对比 |
| 1.4 | ablation 训练脚本 | 多组结构实验指标 | 消融表格和分析 |
| 1.5 | position embedding 版本 | sinusoidal vs learnable 指标 | 位置嵌入差异 |
| 2.1 | sentiment encoder classifier | accuracy、loss、错误案例 | 微调流程和结果 |
| 2.2 | with/without PE 实验 | 两组 accuracy | 词序影响分析 |
| 3.1 | ViT 模型和训练脚本 | ViT accuracy、loss | patch、位置嵌入、分类结果 |
| 3.2 | ResNet/CNN 训练脚本 | CNN accuracy、参数量、耗时 | ViT vs CNN 对比 |
