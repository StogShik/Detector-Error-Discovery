"""Deterministic matching and the compact score composition."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logit

BOX = ['x1', 'y1', 'x2', 'y2']
NAMES = ['person', 'nude_hands', 'cup', 'tray']
SMALL = NAMES[1:]


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_models(root):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    for model in config['models']:
        path = root / model['file']
        if not path.exists() or sha256(path) != model['sha256']:
            raise ValueError(f'Missing or modified model: {path}')
    return config


def overlaps(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(axis=2)
    union = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def assignment(matrix, threshold=.6, soft=False):
    if not matrix.size:
        return {}
    valid = matrix >= threshold
    left, right = linear_sum_assignment(-(valid * (min(matrix.shape) + 1 + matrix)))
    result = {int(i): int(j) for i, j in zip(left, right) if valid[i, j]}
    if soft:
        remaining = [i for i in range(matrix.shape[0]) if i not in result]
        unused = [j for j in range(matrix.shape[1]) if j not in result.values()]
        if remaining and unused:
            left, right = linear_sum_assignment(-matrix[np.ix_(remaining, unused)])
            result.update({remaining[i]: unused[j] for i, j in zip(left, right)})
    return result


def risks(rows, records, names=NAMES, threshold=.1, legacy=False):
    lookup = {r['frame_id']: r for r in records}
    result = pd.Series(index=rows.index, dtype=float)
    for (frame, name), part in rows.groupby(['frame_id', 'class_name']):
        record = lookup[frame]
        boxes = np.asarray(record['boxes'], float).reshape(-1, 4)
        confidence = np.asarray(record['confidence'])
        use = np.asarray(record['classes']) == names.index(name)
        if not legacy:
            use &= confidence >= threshold
        boxes, confidence = boxes[use], confidence[use]
        matrix = overlaps(part[BOX].to_numpy(float), boxes)
        quality = np.zeros(len(part))
        if matrix.size:
            if legacy:
                left, right = linear_sum_assignment(-matrix)
                pairs = {i: j for i, j in zip(left, right) if matrix[i, j] >= .1 and confidence[j] >= .1}
            else:
                pairs = assignment(matrix, soft=True)
            for i, j in pairs.items():
                quality[i] = matrix[i, j]
        result.loc[part.index] = expit((.6 - quality) / .12)
    return result


def nms(boxes, classes, confidence):
    keep = []
    for cls in np.unique(classes):
        pending = np.where(classes == cls)[0][np.argsort(-confidence[classes == cls])]
        while len(pending):
            best = pending[0]
            keep.append(best)
            pending = pending[1:][overlaps(boxes[best:best+1], boxes[pending[1:]])[0] < .5] if len(pending) > 1 else pending[:0]
    return np.asarray(keep, dtype=int)


def tabular_scores(features, root, config):
    from catboost import CatBoostClassifier
    predictions = []
    for model, columns in zip([m for m in config['models'] if m['file'].endswith('.cbm')], config['catboost_columns']):
        estimator = CatBoostClassifier()
        estimator.load_model(str(Path(root) / model['file']))
        predictions.append(estimator.predict_proba(features[columns])[:, 1])
    probability = pd.Series(np.mean(predictions, axis=0), index=features.index)
    score = .45 * (1 - features.confidence).rank(pct=True) + .55 * probability.rank(pct=True)
    clothes = features.p_clothes.fillna(0)
    return score + clothes * (1 - score)


def apply_hand_geometry(score, features, signals, recipe='standard'):
    """Optional rotation-aware candidate; standard remains the shipped default."""
    score = score.copy()
    found = features.class_name.eq('nude_hands') & signals.hand_found.fillna(False).astype(bool)
    risk = signals.hand_risk
    if recipe == 'standard':
        weight = .3
    elif recipe == 'sharp_rotation_confidence':
        if 'hand_consensus' not in signals:
            raise ValueError('The experimental hand recipe requires hand_consensus signals.')
        count = signals.hand_consensus.fillna(0)
        weight = .3 + .2 * ((count - 1) / 3).clip(0, 1)
        weight = weight.loc[found]
        risk = expit(1.5 * logit(risk.clip(1e-8, 1-1e-8)))
    else:
        raise ValueError(f'Unknown hand geometry recipe: {recipe}')
    score.loc[found] = (1-weight) * score.loc[found] + weight * risk.loc[found]
    return score


def compose(features, signals, root, config):
    if features.id.duplicated().any() or signals.id.duplicated().any() or set(features.id) != set(signals.id):
        raise ValueError('Features and signals must contain identical unique detection IDs.')
    signals = signals.set_index('id').loc[features.id].reset_index()
    signals.index = features.index
    old = tabular_scores(features, root, config)
    p = features.p_clothes.fillna(0)
    risk = signals.teacher_risk
    score = .6 * old + .4 * risk + .4 * p * (1 - risk)
    person = features.class_name.eq('person')
    score.loc[person] = .5 * score.loc[person] + .5 * (signals.loc[person, 'grounded_person'] + p.loc[person] * (1 - signals.loc[person, 'grounded_person']))
    for cls, weight in [('nude_hands', .25), ('cup', .5)]:
        mask = features.class_name.eq(cls)
        score.loc[mask] = (1 - weight) * score.loc[mask] + weight * signals.loc[mask, 'grounded_small']
    for _, group in features.loc[features.class_name.eq('tray')].groupby('day'):
        order = (.75 * score.loc[group.index] + .25 * signals.loc[group.index, 'grounded_small']).sort_values(kind='stable').index
        score.loc[order] = np.sort(score.loc[group.index])
    score = apply_hand_geometry(score, features, signals, config.get('hand_geometry', 'standard'))
    if not np.isfinite(score).all() or not score.between(0, 1).all():
        raise ValueError('Scores must be finite values in [0, 1].')
    return features[['id']].assign(score=score)
