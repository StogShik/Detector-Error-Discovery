"""Reproduce the exploratory hand-rule comparison from cached held-out scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from detector_error_discovery.core import apply_hand_geometry, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path, default=ROOT/'reference/hand_recipe_oof.csv')
    parser.add_argument('--output', type=Path, default=Path('outputs/hand_recipe_validation.json')); args = parser.parse_args()
    rows = pd.read_csv(args.input, dtype={'id': str, 'day': str})
    baseline = apply_hand_geometry(rows.score_before_hand, rows, rows)
    candidate = apply_hand_geometry(rows.score_before_hand, rows, rows, 'sharp_rotation_confidence')
    days = {}
    for day, g in rows.groupby('day'):
        a = average_precision_score(g.error, baseline.loc[g.index]); b = average_precision_score(g.error, candidate.loc[g.index])
        days[day] = dict(standard_AP=float(a), candidate_AP=float(b), delta=float(b-a))
    rng = np.random.default_rng(41)
    strata = [[g.index.to_numpy() for _, g in d.groupby('frame_id')] for _, d in rows.groupby('day')]
    deltas = []
    for _ in range(1500):
        values = []
        for groups in strata:
            index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if rows.loc[index, 'error'].nunique() < 2: break
            values.append(average_precision_score(rows.loc[index, 'error'], candidate.loc[index])-average_precision_score(rows.loc[index, 'error'], baseline.loc[index]))
        if len(values) == len(strata): deltas.append(float(np.mean(values)))
    average = float(np.mean([v['standard_AP'] for v in days.values()]))
    candidate_average = float(np.mean([v['candidate_AP'] for v in days.values()]))
    percentiles = np.quantile(deltas, [.025, .5, .975]).tolist()
    passed = candidate_average-average >= .002 and min(v['delta'] for v in days.values()) >= .0005 and percentiles[0] >= 0
    report = dict(rows=len(rows), days=days, standard_mean_AP=average,
        candidate_mean_AP=candidate_average,
        conditional_bootstrap_delta_percentiles=percentiles,
        bootstrap_positive_fraction=float(np.mean(np.asarray(deltas)>0)), input_sha256=sha256(args.input),
        strict_improvement_gate_passed=passed, default_changed=False, public_AP=None,
        limitation='Recipe selected after finite experiments on these same two reused train days. Bootstrap is conditional on selection, not independent evidence. The candidate failed the predeclared .002 mean-gain gate and remains opt-in.')
    args.output.parent.mkdir(exist_ok=True, parents=True); args.output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
