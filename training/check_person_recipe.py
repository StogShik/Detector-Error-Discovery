"""Reproduce the frozen full-body correction on cached held-out TRAIN scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from detector_error_discovery.core import apply_person_geometry, sha256
from detector_error_discovery.person_geometry import evidence

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/person_recipe_validation.json'))
    args = parser.parse_args()
    path = ROOT/'reference/person_recipe_oof.csv'
    rows = pd.read_csv(path, dtype={'id': str, 'day': str})
    frozen = json.loads((ROOT/'reference/person_train_detections.json').read_text())
    recalculated = evidence(rows, frozen)
    np.testing.assert_allclose(rows.person_truncation, recalculated.person_truncation, atol=1e-12, rtol=0)
    report = []
    for baseline in ['baseline_compact', 'baseline_public_model']:
        candidate = apply_person_geometry(rows[baseline], rows, rows)
        other = rows.class_name.ne('person')
        np.testing.assert_array_equal(candidate.loc[other], rows.loc[other, baseline])
        for day, group in rows.groupby('day'):
            for subset in ['all', 'person', 'person_test_like']:
                selected = group if subset == 'all' else group.loc[group.class_name.eq('person')]
                if subset == 'person_test_like':
                    selected = selected.loc[selected.confidence.ge(.73)]
                a = average_precision_score(selected.error, rows.loc[selected.index, baseline])
                b = average_precision_score(selected.error, candidate.loc[selected.index])
                report.append(dict(baseline=baseline, day=day, subset=subset, old_AP=float(a), new_AP=float(b), delta=float(b-a)))
    result = dict(rows=len(rows), person_rows=int(rows.class_name.eq('person').sum()), days=report,
                  input_sha256=sha256(path), default_changed=False, public_AP=None,
                  caveat='Exploratory comparison on two repeatedly reused TRAIN days. No independent Public estimate.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(pd.DataFrame(report).to_string(index=False))


if __name__ == '__main__':
    main()
