"""Optional YOLO26x boundary correction with camera-specific bottom preserved."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
from scipy.optimize import linear_sum_assignment
from .core import overlaps, sha256

MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt'
MODEL_SHA256 = '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'


def download(root):
    path = Path(root)/'.cache/models/yolo26x.pt'
    if path.exists() and sha256(path) == MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Detector-Error-Discovery'})
    with urllib.request.urlopen(request) as source, temporary.open('wb') as target:
        shutil.copyfileobj(source, target)
    if sha256(temporary) != MODEL_SHA256:
        raise ValueError('YOLO26x download checksum mismatch.')
    temporary.replace(path)
    return path


def fuse(old, independent):
    """Keep old count/confidence/bottom; average matched x1,y1,x2 at weight .5."""
    lookup = {r['frame_id']: r for r in independent}
    records = []
    for record in old:
        other = lookup[record['frame_id']]
        boxes = np.asarray(record['boxes'], float).reshape(-1, 4)
        confidence = np.asarray(record['confidence'], float)
        new = np.asarray(other['boxes'], float).reshape(-1, 4)
        scores = np.asarray(other['confidence'], float)
        active = np.flatnonzero(confidence >= .1)
        new = new[scores >= .3]
        matrix = overlaps(boxes[active], new)
        corrected = boxes.copy()
        if matrix.size:
            left, right = linear_sum_assignment(-matrix)
            for local, j in zip(left, right):
                if matrix[local, j] >= .3:
                    i = active[local]
                    corrected[i, :3] = .5*boxes[i, :3]+.5*new[j, :3]
        records.append(dict(frame_id=record['frame_id'], boxes=corrected.tolist(),
                            confidence=confidence.tolist(), classes=np.zeros(len(boxes), int).tolist()))
    return records


def infer(weights, rows, data, cache, device='auto', size=960):
    import torch
    from ultralytics import YOLO
    weights, data, cache = Path(weights), Path(data), Path(cache)
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    frames = sorted(rows.loc[rows.class_name.eq('person'), 'frame_id'].unique())
    signature = hashlib.sha256((sha256(weights)+str(size)+'v1'+device+''.join(frame+sha256(data/f'{frame}.jpg') for frame in frames)).encode()).hexdigest()
    path = cache/f'person_{signature}.json'
    records = json.loads(path.read_text()) if path.exists() else []
    seen = {r['frame_id'] for r in records}
    if seen == set(frames):
        return records
    torch.set_num_threads(3)
    model = YOLO(str(weights))
    assert model.names[0] == 'person'
    cache.mkdir(parents=True, exist_ok=True)
    for number, frame in enumerate(frames, 1):
        if frame in seen:
            continue
        prediction = model.predict(str(data/f'{frame}.jpg'), imgsz=size, conf=.005, iou=.6,
                                   classes=[0], device=device, verbose=False)[0]
        records.append(dict(frame_id=frame, boxes=prediction.boxes.xyxy.cpu().numpy().tolist(),
                            classes=np.zeros(len(prediction.boxes), int).tolist(),
                            confidence=prediction.boxes.conf.cpu().numpy().tolist()))
        path.write_text(json.dumps(records))
        print(f'{weights.stem}: {number}/{len(frames)}', flush=True)
    del model
    import gc
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    return records
