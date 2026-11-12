"""Post-freeze diagnostics and one evaluation of the prespecified source-only follow-up."""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

from experiments.e4.common import (ROOT, RESULTS as BASE, HANDOFF_COLUMNS, load_frame,
    metrics, now, read_json, score_texts, selected_model, setup, sha256, tokenizer,
    write_csv, write_json, AutoModelForSequenceClassification, config, seed_all)
from experiments.e4_followup.train_source import candidate_metrics, policies
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer

OUT = ROOT / 'results/e4_followup'
SPLITS = ['d1_validation', 'd1_test', 'd2_test', 'd3_test', 'd4_test']


def verify_source_selection():
    cfg = read_json(ROOT / 'configs/e4_followup.json')
    frozen = read_json(OUT / 'selection_frozen.json')
    protocol = read_json(OUT / 'protocol_frozen.json')
    assert frozen['config_sha256'] == protocol['config_sha256'] == sha256(ROOT / 'configs/e4_followup.json')
    assert protocol['training_code_sha256'] == sha256(ROOT / 'experiments/e4_followup/train_source.py')
    assert frozen['thresholds_sha256'] == sha256(OUT / 'thresholds_source_only.csv')
    assert cfg['original_checkpoint_sha256'] == sha256(ROOT / 'models/e4/lr2e-5/model.safetensors')
    cp = ROOT / frozen['weighted_checkpoint']
    assert frozen['weighted_checkpoint_sha256'] == sha256(cp / 'model.safetensors')
    thresholds = pd.read_csv(OUT / 'thresholds_source_only.csv', float_precision='round_trip')
    original = pd.read_parquet(BASE / 'handoff/roberta_scores.parquet', filters=[('dataset', '==', 'd1'), ('split', '==', 'validation')])
    weighted = pd.read_csv(OUT / 'weighted_validation_scores.csv', float_precision='round_trip')
    # Recompute choices using only validation scores before accessing held-out outcomes.
    for name, frame in [('original', original), ('benign_weight2', weighted)]:
        expected = policies(frame.gold.to_numpy(), frame.score_attack.to_numpy(), cfg)
        observed = thresholds[thresholds.model == name].reset_index(drop=True)
        for i, row in enumerate(expected):
            assert row['policy'] == observed.iloc[i]['policy']
            assert row['threshold'] == observed.iloc[i]['threshold']
    tuning = pd.read_csv(OUT / 'tuning.csv')
    best = tuning.sort_values(['dev_macro_f1', 'epoch'], ascending=[False, True]).iloc[0]
    assert int(best.epoch) == frozen['weighted_best_epoch']
    winner = 'benign_weight2' if frozen['weighted_dev_macro_f1'] > frozen['original_dev_macro_f1'] else 'original'
    assert winner == frozen['source_selected_model']
    # Compare vectorized threshold enumeration against an independent sklearn implementation.
    for y, s in [(np.array([0, 0, 1, 1]), np.array([.1, .5, .5, .9])),
                 (np.array([0, 1, 1, 0, 0]), np.array([.9, .2, .9, .01, .2]))]:
        for row in candidate_metrics(y, s).itertuples():
            ref = metrics(y, s, row.threshold)
            for key in ['tn', 'fp', 'fn', 'tp', 'fpr', 'macro_f1', 'attack_recall']:
                assert np.isclose(getattr(row, key), ref[key], atol=1e-14)
    return cfg, frozen, cp, thresholds


def diagnose(baseline, frames, cfg):
    joined = {}
    distributions, lengths = [], []
    for name, frame in frames.items():
        dataset, split = name.split('_', 1)
        scores = baseline[(baseline.dataset == dataset) & (baseline.split == split)]
        merged = frame.merge(scores[['sample_id', 'gold', 'score_attack']], on='sample_id', validate='one_to_one')
        assert len(merged) == len(frame) and (merged.gold == merged.label).all()
        merged['pred'] = (merged.score_attack >= .5).astype(int)
        merged['chars'] = merged.text.str.len()
        joined[name] = merged
        for label, group in merged.groupby('label'):
            q = np.quantile(group.score_attack, [0, .05, .25, .5, .75, .95, 1])
            distributions.append({'dataset_split': name, 'gold': int(label), 'n': len(group),
                **dict(zip(['min', 'q05', 'q25', 'median', 'q75', 'q95', 'max'], q))})
            lengths.append({'dataset_split': name, 'gold': int(label), 'n': len(group),
                'chars_median': float(group.chars.median()), 'chars_p95': float(group.chars.quantile(.95)),
                'question_mark_fraction': float(group.text.str.contains(r'\?').mean())})
    write_csv(OUT / 'score_distributions.csv', distributions)
    write_csv(OUT / 'text_characteristics.csv', lengths)
    d3 = joined['d3_test']
    groups = []
    for (label, technique), group in d3.groupby(['label', 'technique'], dropna=False):
        positive = int(group.pred.sum())
        groups.append({'gold': int(label), 'technique': technique, 'n': len(group),
            'predicted_attack': positive, 'positive_rate': positive / len(group),
            'median_score': float(group.score_attack.median())})
    write_csv(OUT / 'd3_technique_rates.csv', groups)
    length_bins = cfg['diagnostics']['character_length_bins']
    length_rates = []
    for name in SPLITS:
        frame = joined[name].query('label == 0').copy()
        frame['length_bin'] = pd.cut(frame.chars, length_bins, right=False)
        for interval, group in frame.groupby('length_bin', observed=True):
            length_rates.append({'dataset_split': name, 'character_bin': str(interval),
                'benign_n': len(group), 'false_positives': int(group.pred.sum()), 'fpr': float(group.pred.mean())})
    write_csv(OUT / 'benign_length_rates.csv', length_rates)
    # Vocabulary and term rankings use D1 train only; external data only characterize coverage.
    train = load_frame('d1_train')
    vec = CountVectorizer(binary=True, lowercase=True, min_df=cfg['diagnostics']['source_lexicon_min_document_frequency'],
                         token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z_]+\b')
    x = vec.fit_transform(train.text)
    pos = np.asarray(x[train.label.to_numpy() == 1].sum(axis=0)).ravel()
    neg = np.asarray(x[train.label.to_numpy() == 0].sum(axis=0)).ravel()
    npos, nneg = int((train.label == 1).sum()), int((train.label == 0).sum())
    a, b = (pos + 1) / (npos + 2), (neg + 1) / (nneg + 2)
    logodds = np.log(a / (1 - a)) - np.log(b / (1 - b))
    terms = pd.DataFrame({'term': vec.get_feature_names_out(), 'attack_documents': pos, 'benign_documents': neg,
                          'source_log_odds': logodds})
    top = terms.sort_values(['source_log_odds', 'term'], ascending=[False, True]).head(cfg['diagnostics']['source_lexicon_top_terms'])
    write_csv(OUT / 'd1_train_attack_associated_terms.csv', top)
    indices = [vec.vocabulary_[term] for term in top.term]
    coverage = []
    for name in SPLITS:
        frame = joined[name]
        counts = np.asarray(vec.transform(frame.text)[:, indices].sum(axis=1)).ravel()
        for gold in [0, 1]:
            for present in [False, True]:
                mask = (frame.label.to_numpy() == gold) & ((counts > 0) == present)
                if mask.sum():
                    coverage.append({'dataset_split': name, 'gold': gold, 'has_source_attack_term': present,
                        'n': int(mask.sum()), 'predicted_attack_rate': float(frame.loc[mask, 'pred'].mean()),
                        'median_score': float(frame.loc[mask, 'score_attack'].median())})
    write_csv(OUT / 'source_lexicon_coverage.csv', coverage)
    # Cases are predetermined by row order and category, not by an attractive masking result.
    cases = d3.query('label == 0 and pred == 1').sort_values('source_row').groupby('technique', sort=True).head(1)
    case_records, mask_records = [], []
    model, tok, _ = selected_model()
    for row in cases.itertuples():
        case_records.append({'sample_id': row.sample_id, 'source_row': int(row.source_row),
            'technique': row.technique, 'text': row.text, 'score_attack': row.score_attack})
        matches = list(re.finditer(r"\b[\w'-]+\b", row.text))
        changed = [row.text[:m.start()] + tok.mask_token + row.text[m.end():] for m in matches]
        scores = score_texts(model, tok, changed)
        for i, (m, score) in enumerate(zip(matches, scores)):
            mask_records.append({'sample_id': row.sample_id, 'position': i, 'word': m.group(),
                'original_score': row.score_attack, 'masked_score': float(score), 'drop': row.score_attack - float(score)})
    del model; gc.collect(); torch.cuda.empty_cache()
    write_csv(OUT / 'd3_diagnostic_cases.csv', case_records)
    write_csv(OUT / 'd3_case_word_masking.csv', mask_records)
    return joined


def evaluate_weighted(cp, selection, frames):
    path = OUT / 'weighted_scores.parquet'
    if path.exists():
        receipt = read_json(OUT / 'external_evaluation_receipt.json')
        assert receipt['scores_sha256'] == sha256(path)
        assert receipt['selection_sha256'] == sha256(OUT / 'selection_frozen.json')
        return pd.read_parquet(path)
    model = AutoModelForSequenceClassification.from_pretrained(cp, local_files_only=True,
        attn_implementation=config()['model']['attn_implementation']).cuda().eval()
    tok = tokenizer(cp)
    rows = []
    for name, frame in frames.items():
        dataset, split = name.split('_', 1)
        score = score_texts(model, tok, frame.text.tolist())
        df = pd.DataFrame({'dataset': dataset, 'split': split, 'sample_id': frame.sample_id, 'gold': frame.label,
            'score_attack': score, 'model_id': 'e4_roberta_benign_weight2_exploratory',
            'checkpoint_rev': selection['weighted_checkpoint_sha256'], 'seed': 6405,
            'transform_id': 'clean', 'parent_id': None})[HANDOFF_COLUMNS]
        rows.append(df)
        if name == 'd1_validation':
            saved = pd.read_csv(OUT / 'weighted_validation_scores.csv', float_precision='round_trip')
            assert np.allclose(score, saved.score_attack, atol=1e-7, rtol=0)
        print('scored', name, len(frame), flush=True)
    result = pd.concat(rows, ignore_index=True)
    assert len(result) == 5086 and not result[['dataset', 'split', 'sample_id']].duplicated().any()
    result.to_parquet(path, index=False)
    write_json(OUT / 'external_evaluation_receipt.json', {'time_utc': now(), 'scores_sha256': sha256(path),
        'selection_sha256': sha256(OUT / 'selection_frozen.json'), 'selection_time_utc': selection['time_utc'],
        'rows': len(result), 'checkpoint_sha256': selection['weighted_checkpoint_sha256']})
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def compare(baseline, weighted, thresholds, selection, cfg):
    rows, intervals = [], []
    for name in SPLITS:
        dataset, split = name.split('_', 1)
        original = baseline[(baseline.dataset == dataset) & (baseline.split == split)].reset_index(drop=True)
        new = weighted[(weighted.dataset == dataset) & (weighted.split == split)].reset_index(drop=True)
        assert original.sample_id.tolist() == new.sample_id.tolist() and original.gold.equals(new.gold)
        for model_name, frame in [('original', original), ('benign_weight2', new)]:
            for threshold in thresholds[thresholds.model == model_name].itertuples():
                rows.append({'model': model_name, 'policy': threshold.policy, 'dataset_split': name,
                    'source_selected_model': model_name == selection['source_selected_model'],
                    **metrics(frame.gold, frame.score_attack, threshold.threshold)})
                for gold, metric in [(0, 'fpr'), (1, 'attack_recall')]:
                    mask = frame.gold.to_numpy() == gold
                    if not mask.any(): continue
                    delta = ((frame.score_attack.to_numpy()[mask] >= threshold.threshold).astype(float)
                             - (original.score_attack.to_numpy()[mask] >= .5).astype(float))
                    rng = np.random.default_rng(cfg['diagnostics']['bootstrap_seed'])
                    samples = rng.choice(delta, size=(cfg['diagnostics']['paired_bootstrap_replicates'], len(delta)), replace=True).mean(axis=1)
                    lo, hi = np.quantile(samples, [.025, .975])
                    intervals.append({'model': model_name, 'policy': threshold.policy, 'dataset_split': name,
                        'metric': metric, 'delta_vs_original_0.5': float(delta.mean()), 'ci025': float(lo), 'ci975': float(hi),
                        'n': len(delta), 'method': 'paired row bootstrap; descriptive; ignores template dependence and selection uncertainty'})
    write_csv(OUT / 'all_policy_metrics.csv', rows)
    write_csv(OUT / 'paired_differences.csv', intervals)


def main():
    setup()
    seed_all(6405)
    cfg, selection, cp, thresholds = verify_source_selection()
    baseline_hash = sha256(BASE / 'handoff/roberta_scores.parquet')
    baseline = pd.read_parquet(BASE / 'handoff/roberta_scores.parquet')
    frames = {name: load_frame(name) for name in SPLITS}
    diagnose(baseline, frames, cfg)
    weighted = evaluate_weighted(cp, selection, frames)
    compare(baseline, weighted, thresholds, selection, cfg)
    assert baseline_hash == sha256(BASE / 'handoff/roberta_scores.parquet')
    verify_source_selection()
    write_json(OUT / 'verification.json', {'time_utc': now(), 'status': 'passed',
        'checks': ['source-only policy recomputation', 'source-only checkpoint and model selection',
            'frozen protocol/config/training code/weights hashes', 'independent threshold-count fixtures with ties',
            'all dataset manifest hashes', 'unique IDs and aligned labels', 'weighted validation inference matches training',
            'original model and score bytes unchanged', 'all eight prespecified policies reported on all five splits'],
        'baseline_scores_sha256': baseline_hash, 'analysis_code_sha256': sha256(Path(__file__))})
    print('Analysis verified.', flush=True)


if __name__ == '__main__': main()
