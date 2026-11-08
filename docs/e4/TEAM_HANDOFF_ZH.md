# E4 完整实验过程与组员交接

## 先看什么

1. `results/e4/report_section.md`：正式 E4 主实验的中英文报告、六组候选训练结果、跨来源参考指标、成本及解释。
2. `results/e4_followup/D3误报分析.html`：独立打开即可阅读的 D3 误报诊断、补充实验、图表与全部结果。
3. `results/e4/presentation_notes_zh.md`：展示讲稿；`experiments/e4/学习与交接说明.md`：方法理解与组员分工。
4. 完整压缩包中的 `START_HERE.html`：合并阅读入口；`MANIFEST.json` 和 `scripts/verify_e4_bundle.py`：离线文件核验。

**正式模型始终为原 E4 的 RoBERTa，学习率 2e-5、epoch 1。D3 补充实验属于探索，未替换原模型。**

## 完整过程

| 阶段 | 实际完成的工作 | 可追溯证据 |
|---|---|---|
| 输入与协议核对 | 使用小组冻结的 6 个数据分割及稳定 ID；核对 E3/E5 提供的材料；只用 D1 train 训练、D1 validation 选择 | `data/audit/`、`results/e4/input_package_provenance.json`、`configs/e4.json` |
| 本机环境 | Windows，RTX 3060 Laptop 6GB；隔离 Python 3.11.9 环境，Torch 2.7.1+cu126，Transformers 4.46.3；实验、缓存、模型均在 E 盘 | `requirements-e4-lock.txt`、`results/e4/run_metadata.json` |
| 资源与更新检查 | 用训练数据进行 batch 4、16 的 smoke test；验证梯度累积与整批更新以及不足整批的尾部样本；选择能容纳且与 E3 一致的 batch 16 | `resource_decision.json`、`smoke_micro*_fp32.json`、`preflight_verification.json` |
| 主训练 | 固定 seed 6405，RoBERTa-base 指定 revision，2e-5 / 5e-5 各训练 3 epochs；FP32 全参数微调，max length 256；两个学习率使用相同初始化和样本顺序 | `training_protocol.json`、`training/`、`tuning.csv`、`run.log`、`events.jsonl`（均位于 `results/e4/`） |
| 选择与冻结 | 从 6 个候选中仅按 D1 validation Macro-F1@0.5 选择 2e-5 / epoch 1；冻结模型、配置和哈希后开展测试评分 | `selected_config.json`、`events.jsonl`、`handoff/` |
| 原始分数交接 | 输出 D1 validation 1985、D1 test 2101、D2 116、D3 545、D4 339，共 5086 行原始分数 | `results/e4/handoff/roberta_scores.parquet` |
| 成本比较 | 同一台电脑复测 E3 DistilBERT 与 E4；固定 20 条预热 + 100 条计时，batch 1，FP32，CPU 4 threads；计时包括分词、传输、前向及分数返回 | `latency_sample_ids.csv`、`latency_trials.csv`、`cost_comparison.csv` |
| 局部解释 | 复用 E2 事先提供的 6 个案例 ID；用 E4 重新推理并做词跨度遮挡；记录局限，未把 E2 解释冒充为 E4 解释 | `occlusion_protocol.json`、`occlusion_cases.csv`、`occlusion_spans.csv`、`case_analysis.md` |
| 主实验核验 | 重新加载选定模型复算全部分数及局部解释，76 项检查通过；检验 E5 分数及成本接口 | `verification.json`、`e5_handoff_compatibility.json` |
| D3 误报诊断 | 分析正常文本分数漂移、4 个正常类别、D1 训练词表覆盖及预定案例遮挡；D3 不进入训练 | `results/e4_followup/` 中的诊断 CSV 与报告 |
| 固定补充实验 | 在新测试推理前固定“正常类损失权重 2”和“D1 validation 召回至少 95% 的最高阈值”；加权模型再训练 3 epochs，按验证集选择 epoch 2 | `configs/e4_followup.json`、`protocol_frozen.json`、`tuning.csv` |
| 补充选择及测试 | 预定的验证指标仍选择原模型；冻结阈值后评估全部 2 模型 × 4 策略 × 5 分割，完整报告恶化结果 | `selection_frozen.json`、`external_evaluation_receipt.json`、`all_policy_metrics.csv`、`verification.json` |

所有日志保留原始时间戳；主要记录为 UTC，换算北京时间加 8 小时。训练纯更新耗时主实验约 564.5 秒、补充训练约 282.3 秒，不含下载、验证和整理报告时间。

## 组员可以直接引用的结论

| 来源 | 原 E4 Macro-F1@0.5 | 攻击召回 | 正常文本误报率 |
|---|---:|---:|---:|
| D1 validation | 0.9863 | 97.56% | 0.38% |
| D1 test | 0.9423 | 89.06% | 0.57% |
| D2 test | 0.8963 | 81.67% | 1.79% |
| D3 test | 0.5073 | 100.00% | 82.64% |
| D4 test（全正常） | 不适用 | 不适用 | 56.34% |

上述为默认 0.5 的参考指标。正式 tF1 / t5 由 E6 从原始分数统一生成，不能直接将本表称为最终小组阈值结果。

D3 正常文本攻击分数中位数 0.9904，远高于 D1 validation 正常文本的 0.00038；D3 AUROC 仍达 0.9624，体现排名与工作阈值的区别。仅用 D1 validation 选择的 R95 阈值 0.9952170848846436 将 D3 FPR 降为 40%，但 D2 攻击召回降到 50%。正常类加权可将 D3 FPR 降为 66.42%，却未通过预定的验证集模型选择。报告必须同时陈述代价。

主模型 GPU 单条推理中位数约 10.78 ms，CPU 约 42.15 ms；同机 E3 约 5.33 / 20.85 ms。主模型 checkpoint 加 tokenizer 约 480 MiB，训练峰值显存约 2634 MiB。单次种子结果不足以证明稳定优势。

## 分工交接

| 接收人 | 需要的文件 / 后续工作 |
|---|---|
| E6 / 统一评测 | `results/e4/handoff/`：保留 stable ID、seed、checkpoint hash；从 D1 validation 选正式阈值，再统一算指标。补充实验分数有单独 model_id，不替代主分数。 |
| E5 / 成本分析 | 主分数、`results/e4/latency.csv`、`model_size.csv`、`cost_comparison.csv`；E4 已记录接口核验结果。 |
| 应用整合 | `models/e4/lr2e-5/` + `experiments/e4/predict.py`；使用 E6 后续冻结的阈值。 |
| 报告 / 视频 | 主报告、D3 补充报告、两套 figures、英文段落、中文讲稿；明确区分主实验和探索。 |
| 数据 / E3 同学 | 完整包携带已有冻结数据、E1/E3 分数及 E3 选定模型，仅为复现比较；这些支持材料不是 E4 的原创贡献。 |

## 包的内容与复现

**阅读版 ZIP** 包含代码、参数、全部报告图表、过程日志、原始分数、冻结数据、依赖清单及模型 tokenizer/config；不含三个大型权重文件，适合群内阅读与审阅。

**完整复现版 ZIP** 额外包含三个权重：正式 E4、未选中的探索性加权 E4，以及组员提供的 E3 最佳模型（用于同机成本复测）。没有打包虚拟环境、下载缓存、优化器恢复快照或未选中的主实验 5e-5 权重；六个主候选和三个补充候选的训练过程与选择证据均完整保留。从头重复训练需下载固定 revision 的预训练模型并安装锁定依赖。

先解压，然后在包根目录执行 `python scripts/verify_e4_bundle.py`。此命令仅依赖 Python 标准库，会核对清单中的每个文件，包括完整包中的权重。阅读版会清楚提示哪些大权重未包含。

安装和复现步骤见 `experiments/e4/README.md` 与 `experiments/e4_followup/README.md`。已有结果不可用训练命令覆盖；独立重复训练应使用新目录和新的缓存。`run_e4.ps1` 的 `RuntimeRoot` 可指定本机环境位置。不要因跨机器推理时间不同而改写本次已测成本结果。

## Git 范围、来源与学术说明

Git 提交仅包含 E4 / E4 follow-up 的代码、参数、小型证据、文档、打包脚本及必要的 tokenizer/config。E3/E5 支持文件仅随完整材料包附带，不作为 E4 改动提交；大权重由压缩包传递，不直接提交 Git。源仓库为 `Run-Y/promptshield`，E4 开始时基线提交 `031b119918143e8048518b5bb6037d299b283b5f`。

本实验与报告使用 AI 辅助编写、运行和整理。组员应理解并核查自己提交的部分，在课程 Academic Integrity and AI Disclosure Form 中如实披露；包内不伪造个人签名、教师批准或独立原创声明。D3 补充实验是看到旧测试结果后提出的探索，不能宣传成新的盲测。所有数值选择只使用 D1 开发数据。
