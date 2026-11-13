"""Frozen CPU hand landmarks and geometric hand-box risk."""
import json
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.special import expit
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def inverse(points, k, size):
    x, y = (points[:, 0].copy(), points[:, 1].copy())
    end = size - 1
    if k == 1:
        return np.column_stack([end - y, x])
    if k == 2:
        return np.column_stack([end - x, end - y])
    if k == 3:
        return np.column_stack([y, end - x])
    return points

def extract(rows, root, asset, destination):
    rows = rows.loc[rows.class_name.eq('nude_hands')]
    records = json.loads(destination.read_text()) if destination.exists() else []
    seen = {r['id'] for r in records}
    if seen == set(rows.id):
        return records
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(asset), delegate=python.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.IMAGE, num_hands=4, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.5)
    with vision.HandLandmarker.create_from_options(options) as model:
        for number, row in enumerate(rows.itertuples(), 1):
            if row.id in seen:
                continue
            start = time.monotonic()
            image = cv2.imread(str(root / f'{row.frame_id}.jpg'))
            if image.shape[:2] != (720, 1280):
                image = cv2.resize(image, (1280, 720))
            a = np.array([row.x1, row.y1, row.x2, row.y2])
            center = (a[:2] + a[2:]) / 2
            half = np.maximum((a[2:] - a[:2]) * 3, 64)
            bounds = np.r_[np.floor(center - half), np.ceil(center + half)].astype(int)
            bounds = np.clip(bounds, 0, [1280, 720, 1280, 720])
            x1, y1, x2, y2 = bounds
            crop = image[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            scale = 512 / max(w, h)
            rw, rh = (round(w * scale), round(h * scale))
            left, top = ((512 - rw) // 2, (512 - rh) // 2)
            patch = np.full((512, 512, 3), 114, np.uint8)
            patch[top:top + rh, left:left + rw] = cv2.resize(crop, (rw, rh))
            hands = []
            for k in [0, 1, 2, 3]:
                rotated = np.ascontiguousarray(np.rot90(patch, k))
                im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
                result = model.detect(im)
                for points, label in zip(result.hand_landmarks, result.handedness):
                    xy = np.array([[p.x * 512, p.y * 512] for p in points])
                    xy = inverse(xy, k, 512)
                    xy = (xy - [left, top]) / scale + [x1, y1]
                    hands.append(dict(points=xy.tolist(), handedness_score=float(label[0].score), rotation=90 * k))
            records.append(dict(id=row.id, frame_id=row.frame_id, bounds=bounds.tolist(), hands=hands))
            destination.write_text(json.dumps(records))
            if number % 10 == 0:
                print(f'HANDS target {number}/{len(rows)} hands={len(hands)} time={time.monotonic() - start:.2f}s', flush=True)
    return records

def summarize(rows, records):
    lookup = {r['id']: r for r in records}
    result = []
    for r in rows.itertuples():
        a = np.array([r.x1, r.y1, r.x2, r.y2])
        wh = a[2:] - a[:2]
        center = (a[:2] + a[2:]) / 2
        qualities = []
        coverage = []
        detections = []
        for hand in lookup[r.id]['hands']:
            points = np.asarray(hand['points'])
            dist = np.linalg.norm((points[0] - center) / wh)
            if dist > 2:
                continue
            lo, hi = (points.min(axis=0), points.max(axis=0))
            size = hi - lo
            if size.min() < 3:
                continue
            b = np.r_[lo - 0.12 * size, hi + 0.12 * size]
            intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
            iou = intersection / max(1, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection)
            qualities.append(iou)
            coverage.append(np.mean(((points >= a[:2]) & (points <= a[2:])).all(axis=1)))
            detections.append((b, hand['rotation']))
        consensus = 0
        for box, rotation in detections:
            rotations = {rotation}
            for other, k in detections:
                intersection = np.maximum(np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]), 0).prod()
                overlap = intersection / max(1, (box[2:] - box[:2]).prod() + (other[2:] - other[:2]).prod() - intersection)
                if overlap >= 0.5:
                    rotations.add(k)
            consensus = max(consensus, len(rotations))
        q = max(qualities, default=0)
        found = len(qualities) > 0
        result.append(dict(id=r.id, hand_found=found, hand_iou=q, hand_coverage=max(coverage, default=0), consensus=consensus, risk=float(expit((0.6 - q) / 0.12))))
    return pd.DataFrame(result)
