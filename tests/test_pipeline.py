from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from detector_error_discovery.core import assignment, compose, verify_models

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
