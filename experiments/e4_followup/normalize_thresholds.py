"""Lossless CSV parsing correction, using validation only and before external inference."""
import shutil
import pandas as pd
from experiments.e4.common import ROOT, RESULTS, read_json, sha256, write_csv, write_json, now
from experiments.e4_followup.train_source import policies, OUT, CFG


def main():
    assert not (OUT / 'weighted_scores.parquet').exists(), 'Never adjust after external evaluation.'
    assert not (OUT / 'external_evaluation_receipt.json').exists()
    receipt = OUT / 'precision_correction.json'
    if receipt.exists():
        print('Already normalized; no mutation.')
        return
    cfg = read_json(CFG)
    selection = read_json(OUT / 'selection_frozen.json')
    old = pd.read_csv(OUT / 'thresholds_source_only.csv', float_precision='round_trip')
    baseline = pd.read_parquet(RESULTS / 'handoff/roberta_scores.parquet', filters=[('dataset', '==', 'd1'), ('split', '==', 'validation')])
    weighted = pd.read_csv(OUT / 'weighted_validation_scores.csv', float_precision='round_trip')
    rows = []
    for name, df in [('original', baseline), ('benign_weight2', weighted)]:
        rows.extend({'model': name, **r} for r in policies(df.gold.to_numpy(), df.score_attack.to_numpy(), cfg))
    new = pd.DataFrame(rows)
    assert old[['model', 'policy']].equals(new[['model', 'policy']])
    max_delta = float(abs(old.threshold - new.threshold).max())
    assert max_delta < 1e-14
    for key in ['tp', 'fp', 'tn', 'fn']: assert old[key].equals(new[key])
    for filename in ['thresholds_source_only.csv', 'selection_frozen.json']:
        shutil.copy2(OUT / filename, OUT / (filename + '.before_precision_fix'))
    write_csv(OUT / 'thresholds_source_only.csv', new)
    selection['initial_freeze_time_utc'] = selection['time_utc']
    selection['time_utc'] = now()
    selection['thresholds_sha256'] = sha256(OUT / 'thresholds_source_only.csv')
    selection['precision_correction'] = 'Lossless CSV round-trip parsing before external evaluation; same policy, models and validation confusion matrices.'
    write_json(OUT / 'selection_frozen.json', selection)
    write_json(receipt, {'time_utc': now(), 'max_threshold_change': max_delta,
        'reason': 'Default pandas float parser changed a boundary score at about 1e-16; preserve exact float32 inference scores.',
        'data_read': 'D1 validation only', 'validation_confusion_counts_changed': False,
        'code_sha256': sha256(ROOT / 'experiments/e4_followup/normalize_thresholds.py')})
    print('Normalized', max_delta)


if __name__ == '__main__': main()
