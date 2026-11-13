"""Label-free one-to-one neighbor and duplicate features."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from . import scene_features as scene
TEMPORAL = ['temporal_available', 'temporal_matches', 'temporal_both', 'temporal_mean_iou', 'temporal_confidence_delta', 'temporal_center_shift', 'temporal_size_change', 'temporal_stationary']
DUPLICATE = ['duplicate_pressure', 'duplicate_confidence_gap']

def add_features(rows, split, root):
    rows = rows.copy()
    stream = pd.read_csv(root / f'{split}_stream_predictions.csv', dtype={'detection_id': str})
    groups = {key: group for key, group in stream.groupby(['frame_id', 'class_name'])}
    context = pd.read_csv(root / f'{split}_context.csv').set_index('frame_id')
    assert context.index.is_unique
    rows[DUPLICATE + TEMPORAL] = 0.0
    for (frame_id, class_name), targets in rows.groupby(['frame_id', 'class_name']):
        current = groups[frame_id, class_name]
        boxes = current[scene.BOX].to_numpy(float)
        confidence = current.confidence.to_numpy()
        positions = {value: i for i, value in enumerate(current.detection_id)}
        for index, row in targets.iterrows():
            overlaps = scene.intersection_over_union(row[scene.BOX].to_numpy(float), boxes)
            higher = confidence > row.confidence
            pressure = overlaps * higher * confidence
            best = pressure.argmax()
            rows.loc[index, DUPLICATE] = [pressure[best], max(confidence[best] - row.confidence, 0) if higher[best] else 0]
        matches = {value: [] for value in targets.id}
        available = 0
        for direction in ['previous', 'next']:
            neighbor = context.loc[frame_id, direction]
            if pd.isna(neighbor):
                continue
            available += 1
            candidates = groups.get((neighbor, class_name))
            if candidates is None or candidates.empty:
                continue
            other = candidates[scene.BOX].to_numpy(float)
            ious = np.vstack([scene.intersection_over_union(box, other) for box in boxes])
            valid = ious >= 0.3
            left, right = linear_sum_assignment(-(valid.astype(float) * (min(ious.shape) + 1) + ious * valid))
            assignment = {i: j for i, j in zip(left, right) if valid[i, j]}
            for _, row in targets.iterrows():
                i = positions[row.id]
                if i not in assignment:
                    continue
                j = assignment[i]
                size = np.maximum(boxes[i, 2:] - boxes[i, :2], 1.0)
                other_size = np.maximum(other[j, 2:] - other[j, :2], 1.0)
                shift = np.linalg.norm((boxes[i, :2] + boxes[i, 2:] - (other[j, :2] + other[j, 2:])) / 2 / size)
                change = np.abs(np.log(other_size / size)).mean()
                matches[row.id].append((ious[i, j], candidates.confidence.iloc[j] - row.confidence, shift, change))
        for index, row in targets.iterrows():
            matched = matches[row.id]
            mean = np.mean(matched, axis=0) if matched else np.zeros(4)
            rows.loc[index, TEMPORAL] = [available, len(matched), float(len(matched) == 2), *mean, float(bool(matched) and mean[2] < 0.1 and (mean[3] < 0.1))]
    assert np.isfinite(rows[DUPLICATE + TEMPORAL].to_numpy()).all()
    return rows
