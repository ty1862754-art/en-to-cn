# 报告与提交检查清单

## 报告必填项

- [ ] 写清楚 Encoder 与 Decoder 的结构差异和功能差异。
- [ ] 写清楚 padding mask 和 causal mask 的实现与动机。
- [ ] 写清楚 Decoder 训练阶段和推理阶段的差异。
- [ ] 插入机器翻译训练 loss 曲线。
- [ ] 报告验证集 BLEU。
- [ ] 报告测试集 BLEU。
- [ ] 列出若干测试集翻译示例并分析。
- [ ] 记录 warm-up cosine 学习率调优结果。
- [ ] 用表格记录超参数消融实验。
- [ ] 实现并讨论新的位置嵌入方式。
- [ ] 情感分析加载第一部分 encoder 预训练参数。
- [ ] 插入情感分析 train loss 曲线。
- [ ] 插入情感分析 validation accuracy 曲线。
- [ ] 比较情感分析中有/无位置嵌入的性能。
- [ ] 实现 CIFAR10 ViT。
- [ ] 插入 ViT train loss 曲线和 validation accuracy 曲线。
- [ ] 实现 ResNet50/CNN 对比实验。
- [ ] 讨论 ViT 相较 CNN 的优势与不足。
- [ ] 写完整代码运行说明。

## 文件夹建议

- `report/main.tex`：LaTeX 报告主文件。
- `assets/`：保存 loss curve、accuracy curve、实验截图、表格图片。
- `code_notes/`：保存每个任务的运行命令、参数记录、实验日志摘要。
- `assignment_completion_plan.md`：从 PDF 提取的完成计划。
- `report_checklist.md`：最终提交前逐项核对。

## 提交前检查

- [ ] ZIP 文件名符合 `学号_assignment2.zip`。
- [ ] ZIP 中包含报告 PDF 或 LaTeX 源文件。
- [ ] ZIP 中包含所有代码。
- [ ] ZIP 中包含运行说明。
- [ ] ZIP 中说明大文件/数据集路径或提供必要文件。
- [ ] 所有图表在报告中能正常显示。
- [ ] 所有表格中的结果都有对应实验产物支撑。
- [ ] 报告中的代码路径与实际文件路径一致。

