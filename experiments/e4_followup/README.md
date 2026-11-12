# D3 false-positive follow-up (exploratory)

This add-on preserves the original E4 checkpoint and results. Open
`results/e4_followup/D3误报分析.html` for the self-contained Chinese report and figures.
The same content, including an English report paragraph, is in
`results/e4_followup/D3误报诊断与改进实验.md`.

## Decision and scientific status

- Retain the original E4 model: its D1 validation Macro-F1 at 0.5 exceeds that of the
  fixed benign-loss-weight-2 candidate (0.986348 vs 0.983813).
- Original-model D1-only R95 threshold is 0.9952170848846436. It reduces D3 FPR from
  82.64% to 40.00%, but D2 attack recall falls from 81.67% to 50.00%. It is an
  exploratory operating policy, not a recommended universal default.
- This method was motivated by previously observed test behavior. All fitting is
  D1 train only; all numerical choices are D1 validation only. Do not call D3 a
  fresh blind confirmation. Do not select a model or policy by the new D3 scores.
- `tF1_reference` and `t5_reference` are supplementary calculations; E6 retains
  ownership of the official operating thresholds. D4 has no attack examples.

## Integration and reproduction

The small ZIP is a project-root overlay for the previously delivered
`PromptShield_E4` project. It only adds `configs/e4_followup.json`,
`experiments/e4_followup/`, and `results/e4_followup/`. The optional model ZIP adds
`models/e4_followup/benign_weight2/`. It does not replace `models/e4/`.
Use the original E4 Python environment and dependencies.

```powershell
$env:E4_CACHE = 'E:\EE6405_E4\cache'
$env:TEMP = 'E:\EE6405_E4\tmp'
$env:TMP = $env:TEMP
& 'E:\EE6405_E4\.venv\Scripts\python.exe' -m experiments.e4_followup.analyze
& 'E:\EE6405_E4\.venv\Scripts\python.exe' -m experiments.e4_followup.report
```

Run from the integrated project root. `analyze` checks all frozen hashes,
recomputes selection using D1 validation, regenerates diagnostics, and uses the
cached weighted-model scores if the receipt is valid. It does not tune new
parameters or change selection. `report` rebuilds the figures and reports.

For a fully independent training reproduction, use a new isolated checkout and
output directory containing no prior follow-up results. Run, in order:

```text
python -m experiments.e4_followup.train_source
python -m experiments.e4_followup.normalize_thresholds
python -m experiments.e4_followup.analyze
python -m experiments.e4_followup.report
```

The training command refuses to overwrite frozen selection. Do not delete that
guard to keep searching against the already inspected tests. Normalization is a
lossless CSV-boundary correction performed before external inference, with the
same rules and validation confusion counts; its audit record is retained.

## Audit and files

- `protocol_frozen.json`: training/config hash, seed, environment, and rules.
- `selection_frozen.json`: validation-selected epoch/model, checkpoint and policy hashes.
- `external_evaluation_receipt.json`: later evaluation time, selection and score hashes.
- `verification.json`: independent count fixtures, dataset alignment, validation
  prediction reproduction, frozen choices, and unchanged baseline bytes.
- `thresholds_source_only.csv`: use full floating-point precision (`float_precision='round_trip'`
  in pandas); classify with `score_attack >= threshold`.
- `all_policy_metrics.csv`: all 40 model/policy/split combinations, no hidden failures.
- `weighted_scores.parquet`: 5,086 rows in the shared score contract; distinct
  `model_id=e4_roberta_benign_weight2_exploratory`.
- `d3_diagnostic_cases.csv`, `d3_case_word_masking.csv`, source lexicon and length
  tables: diagnosis only, not training examples or causal proof.
- `paired_differences.csv`: descriptive row-bootstrap intervals, not accounting
  for template clustering or method-selection uncertainty.

Training and report preparation used AI assistance. The submitting student should
review and understand the implementation and disclose the assistance accurately.
