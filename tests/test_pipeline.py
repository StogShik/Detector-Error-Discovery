from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from detector_error_discovery.core import apply_person_geometry, assignment, compose, verify_models

ROOT = Path(__file__).resolve().parents[1]


def test_cardinality_precedes_iou_sum():
    matrix = np.array([[.9, .61], [.6, .59]])
    # Maximizing IoU sum alone would incorrectly return only one valid pair.
    assert assignment(matrix) == {0: 1, 1: 0}


def test_empty_matching():
    assert assignment(np.empty((2, 0))) == {}


def test_reproduces_compact_reference_and_aligns_ids():
    config = verify_models(ROOT)
    features = pd.read_csv(ROOT/'reference/features.csv', dtype={'id': str})
    signals = pd.read_csv(ROOT/'reference/signals.csv', dtype={'id': str}).sample(frac=1, random_state=7)
    expected = pd.read_csv(ROOT/'reference/submission.csv', dtype={'id': str}).set_index('id')
    actual = compose(features, signals, ROOT, config).set_index('id')
    assert set(actual.index) == set(expected.index)
    np.testing.assert_allclose(actual.loc[expected.index, 'score'], expected.score, rtol=0, atol=1e-12)
    assert actual.score.between(0, 1).all()


def test_rejects_duplicated_ids():
    features = pd.read_csv(ROOT/'reference/features.csv', dtype={'id': str})
    signals = pd.read_csv(ROOT/'reference/signals.csv', dtype={'id': str})
    with pytest.raises(ValueError, match='unique detection IDs'):
        compose(features, pd.concat([signals, signals.iloc[:1]]), ROOT, json.loads((ROOT/'config.json').read_text()))


def test_experimental_hand_recipe_matches_fixture_and_preserves_other_classes():
    config = verify_models(ROOT)
    features = pd.read_csv(ROOT/'reference/features.csv', dtype={'id': str})
    signals = pd.read_csv(ROOT/'reference/signals.csv', dtype={'id': str})
    baseline = compose(features, signals, ROOT, config).set_index('id')
    actual = compose(features, signals, ROOT, dict(config, hand_geometry='sharp_rotation_confidence')).set_index('id')
    expected = pd.read_csv(ROOT/'reference/submission_hand_consensus.csv', dtype={'id': str}).set_index('id')
    np.testing.assert_allclose(actual.loc[expected.index, 'score'], expected.score, atol=1e-12, rtol=0)
    untouched = features.loc[features.class_name.ne('nude_hands'), 'id']
    np.testing.assert_array_equal(actual.loc[untouched, 'score'], baseline.loc[untouched, 'score'])


def test_experimental_hand_recipe_requires_consensus():
    config = verify_models(ROOT)
    features = pd.read_csv(ROOT/'reference/features.csv', dtype={'id': str})
    signals = pd.read_csv(ROOT/'reference/signals.csv', dtype={'id': str}).drop(columns='hand_consensus')
    with pytest.raises(ValueError, match='requires hand_consensus'):
        compose(features, signals, ROOT, dict(config, hand_geometry='sharp_rotation_confidence'))


def test_fullbody_evidence_distinguishes_truncation_from_neighbor():
    from detector_error_discovery.person_geometry import evidence
    rows = pd.DataFrame(dict(id=['cut', 'full', 'neighbor', 'cup'], frame_id=['f']*4,
                             class_name=['person', 'person', 'person', 'cup'],
                             x1=[100, 100, 500, 100], y1=[100]*4,
                             x2=[200, 200, 600, 200], y2=[250, 400, 250, 250]))
    raw = [dict(frame_id='f', boxes=[[100, 100, 200, 400]], confidence=[.95])]
    signal = evidence(rows, raw).person_truncation
    assert signal.iloc[0] > .95
    assert signal.iloc[1] < .03
    assert signal.iloc[2] == 0
    assert signal.iloc[3] == 0


def test_person_recipe_reproduces_both_bases_and_preserves_other_classes():
    features = pd.read_csv(ROOT/'reference/features.csv', dtype={'id': str})
    signal = pd.read_csv(ROOT/'reference/person_truncation_signals.csv', dtype={'id': str}).set_index('id').loc[features.id].reset_index()
    for base, expected in [('submission.csv', 'submission_person_truncation.csv'),
                           ('submission_public_087853.csv', 'submission_person_truncation_publicbase.csv')]:
        original = pd.read_csv(ROOT/'reference'/base, dtype={'id': str}).set_index('id')
        old = features.id.map(original.score)
        new = apply_person_geometry(old, features, signal)
        target = features.id.map(pd.read_csv(ROOT/'reference'/expected, dtype={'id': str}).set_index('id').score)
        np.testing.assert_allclose(new, target, rtol=0, atol=1e-12)
        other = features.class_name.ne('person')
        np.testing.assert_array_equal(old.loc[other], new.loc[other])
        assert new.ge(old).all()
