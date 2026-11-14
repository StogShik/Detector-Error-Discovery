"""Reproduce the experimental YOLO26x-fusion submission on competition TEST.

Earlier full-graph model outputs are cached and fixed. --data-dir regenerates
both person detector outputs from actual JPEGs; other branches stay cached.
This command is for the original competition TEST, not arbitrary new footage.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from detector_error_discovery.core import BOX, risks, sha256, verify_models
from detector_error_discovery.strong_person import download, fuse, infer
from train_diverse_person import graph

ROOT = Path(__file__).resolve().parents[1]


def correction(rows, old, new, offset=.05):
    person = rows.class_name.eq('person')
    original_risk = risks(rows.loc[person], old, names=['person'], legacy=True)
    np.testing.assert_allclose(original_risk, rows.loc[person, 'teacher_risk'], atol=1e-12, rtol=0)
    fused = risks(rows.loc[person], fuse(old, new), names=['person'], legacy=True)
    updated = rows.baseline.copy()
    delta = graph(rows, fused)-graph(rows)
    updated.loc[person] += delta.loc[person]
    np.testing.assert_array_equal(updated.loc[~person], rows.loc[~person, 'baseline'])
    scores = (updated+offset*person)/(1+offset)
    assert np.isfinite(scores).all() and scores.between(0, 1).all()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Regenerate person detections from the original TEST JPEGs')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/submission_yolo26x_fusion.csv')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'.cache/strong_person')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    rows = pd.read_csv(ROOT/'reference/strong_person_test_graph_inputs.csv', dtype={'id': str, 'day': str})
    if args.data_dir is None:
        old = json.loads((ROOT/'reference/strong_person_old_test.json').read_text())
        new = json.loads((ROOT/'reference/strong_person_yolo26x_test.json').read_text())
    else:
        verify_models(ROOT)
        actual = pd.read_csv(args.data_dir/'test.csv', dtype={'id': str}).set_index('id')
        assert set(actual.loc[actual.class_name.ne('table')].index) == set(rows.id)
        np.testing.assert_array_equal(actual.loc[rows.id, 'frame_id'].to_numpy(), rows.frame_id.to_numpy())
        np.testing.assert_array_equal(actual.loc[rows.id, 'class_name'].to_numpy(), rows.class_name.to_numpy())
        np.testing.assert_allclose(actual.loc[rows.id, BOX].to_numpy(), rows[BOX].to_numpy(), atol=1e-8, rtol=0)
        np.testing.assert_allclose(actual.loc[rows.id, 'confidence'].to_numpy(), rows.confidence.to_numpy(), atol=1e-12, rtol=0)
        old = infer(ROOT/'models/person.pt', rows, args.data_dir, args.cache_dir, args.device, size=1280)
        new = infer(download(ROOT), rows, args.data_dir, args.cache_dir, args.device, size=960)
    score = correction(rows, old, new)
    base_path = ROOT/'reference/submission_public_087853.csv'
    assert sha256(base_path) == '9a01485aca01a2934ab057c8f12a257ecfb991d3f5ce7cffa973fb5e7458c352'
    base = pd.read_csv(base_path, dtype={'id': str}).set_index('id')
    np.testing.assert_allclose(rows.baseline, base.loc[rows.id, 'score'], atol=1e-12, rtol=0)
    template = base.reset_index()[['id']]
    result = template.merge(rows[['id']].assign(score=score), on='id', validate='one_to_one', sort=False)
    expected = pd.read_csv(ROOT/'reference/submission_yolo26x_fusion.csv', dtype={'id': str})
    assert len(result) == 362 and result.id.equals(expected.id)
    np.testing.assert_allclose(result.score, expected.score, atol=1e-12, rtol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Wrote {len(result)} rows to {args.output}. Recorded Public AP: 0.88545 (user leaderboard screenshot).')


if __name__ == '__main__':
    main()
