# E4 RoBERTa learning rate cost and local explanations

This experiment fine-tunes RoBERTa-base on the frozen D1 training split, selects a checkpoint using only D1 validation, and exports raw attack scores for E6. It also measures inference cost and word-span masking sensitivity. All reported scores must come from the saved checkpoint and pass `verify.py`.

## Protocol

- Model: `FacebookAI/roberta-base`, revision recorded in `configs/e4.json`.
- Seed 6405; learning rates 2e-5 and 5e-5; three epochs for each rate.
- Maximum sequence length 256, dynamic padding, full-parameter FP32 fine-tuning, physical and effective batch size 16.
- AdamW, weight decay 0.01 excluding bias and LayerNorm weights, betas (0.9, 0.999), epsilon 1e-8; 10% linear warm-up then linear decay; gradient norm clipped at 1.0.
- The same initial model weights, random seed and epoch permutations are used for both rates. The final incomplete batch is retained. No extra seeds are claimed.
- Choose from six (learning rate, epoch) candidates using D1 validation Macro F1 at threshold 0.5; ties prefer an earlier epoch, then a smaller learning rate.
- Freeze the checkpoint and hashes before reading any test split for evaluation. E6 owns the final tF1 and t5 policies. E4's threshold-0.5 tables are clearly marked reference-only.
- D4 is an optional additional raw-score handoff for E6; it is benign-only and has no F1, attack recall or AUROC result.

The supplied E3 protocol has the same primary seed, learning-rate grid, epoch budget, maximum length, batch, precision, optimizer, schedule and selection criterion. E3 used Transformers 5.22.2 / Torch 2.7.1+cu128; this E4 environment uses Transformers 4.46.3 / Torch 2.7.1+cu126. Different backbones, tokenizers, pretraining, libraries and devices prevent a single-factor architecture claim. E3's supplied checkpoint is remeasured on the E4 machine for the cost comparison.

## Run

Use an isolated Python 3.11 environment; do not install over the user's global environment. On this machine the environment is `E:\EE6405_E4\.venv` and every large cache is on E:. Install PyTorch from its official CUDA wheel index, then install the requirements. `requirements-e4-lock.txt` records the complete environment used for the run.

```powershell
python -m venv E:\EE6405_E4\.venv
E:\EE6405_E4\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
E:\EE6405_E4\.venv\Scripts\python.exe -m pip install -r requirements-e4.txt
$env:E4_CACHE = 'E:\EE6405_E4\cache'
$env:TEMP = 'E:\EE6405_E4\tmp'
$env:TMP = $env:TEMP
```

From a fresh project checkout with the six frozen `data/processed` files and audit manifest:

```powershell
python -m experiments.e4.verify --preflight
python -m experiments.e4.train smoke --micro-batch 16 --precision fp32 --steps 50
python -m experiments.e4.train train
python -m experiments.e4.evaluate score
python -m experiments.e4.evaluate benchmark
python -m experiments.e4.evaluate explain
python -m experiments.e4.figures
python -m experiments.e4.verify
python -m experiments.e4.report
```

`python` in these commands must be the isolated environment's executable. The delivered package already contains results: run the verify, figures or inference commands directly. To repeat training independently, use a fresh checkout containing the E4 source/configuration and frozen data, and set `E4_CACHE` to a new directory such as `E:\EE6405_E4\reproduction_cache` so previous resume snapshots are not reused. The training and score commands deliberately refuse to silently replace a completed selection or raw-score file. Interrupted training resumes from the last completed epoch using cache snapshots and checked protocol hashes; partial-epoch step logs are diagnostic only.

## Handoff

| Recipient | Files |
|---|---|
| E6 | `results/e4/handoff/roberta_scores.parquet`, selected configuration, all six tuning rows, run metadata, selected checkpoint and tokenizer |
| E5 | Raw scores above, `latency.csv`, `model_size.csv`, `selected_config.json`, `comparison_at_0.5.csv`; final group metrics must come from E6 |
| E3 app owner | Selected checkpoint, `experiments/e4/predict.py`, and E6's later frozen thresholds |
| E2 case owner | `occlusion_cases.csv`, `occlusion_spans.csv`, the occlusion figure and limitations |
| Report/video owner | `report_section.md`, `findings_and_limitations.md`, figures, run commands and evidence manifest |

The raw-score file has 5,086 rows: 1,985 D1 validation, 2,101 D1 test, 116 D2 test, 545 D3 test and 339 D4 test. It preserves stable IDs and the ten-column README contract; clean rows have `transform_id=clean` and a null parent. Scores increase toward class 1 (attack), but are not calibrated risks. Checkpoint identifiers are SHA-256 digests of `model.safetensors`.

The checkpoint stays in `models/e4/lr...`; `checkpoint_path.txt` contains a complete CPU loading example. Model weights are too large for ordinary GitHub file uploads. Share the separately packaged checkpoint with teammates and commit only the E4 code/configuration/small evidence using your team's normal workflow.

## Cost and explanation scope

Latency includes tokenization, host-to-device transfer, forward pass, softmax and returning a CPU scalar. It excludes loading the model. Both encoders use the same 20 warm-up and 100 measured D1 validation IDs, batch 1, FP32, four CPU threads, and the same machine. Record GPU allocation separately from reservation; training peak and inference peak are different measurements. `cost_comparison.csv` contains both encoders; `latency.csv` contains E4 only for clean integration with E5.

Six case IDs are fixed from E2's supplied case list before E4 predictions are observed. Their gold labels and text are re-read from the original processed data. Each visible whitespace-delimited span is mapped to its BPE tokens, which are replaced with `<mask>` tokens without changing sequence positions or length. Positive delta means the original attack score decreases after masking. Negative delta means it increases. These are local sensitivity measurements; mask artifacts, interactions and case selection prevent causal or representative-error claims. E2's predictions and explanations are not copied as RoBERTa findings.

## Provenance and human review

`training_protocol.json`, `events.jsonl`, hashes, run logs, the environment lock and verification results provide the experimental record. This implementation and writing were developed with AI assistance and executed on the user's computer. They do not certify independent student authorship or human verification. The student should inspect the method and cases, make and record their own decisions, and accurately complete the course's AI disclosure and contribution forms. No signature or instructor approval is supplied by this package.

References: [RoBERTa model card](https://huggingface.co/FacebookAI/roberta-base), [versioned model API](https://huggingface.co/docs/transformers/v4.46.3/model_doc/roberta), [PyTorch version installation instructions](https://pytorch.org/get-started/previous-versions/), supplied PromptShield plan and repository handoff README.
