"""Optional full-body evidence for person boxes truncated below the torso."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .core import BOX, sha256

MODEL_URL = 'https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view'
MODEL_SHA256 = 'e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba'


def evidence(rows, records):
    """Return per-proposal lower-body evidence without changing other classes."""
    result = pd.DataFrame({'id': rows.id, 'person_truncation': 0.}, index=rows.index)
    lookup = {record['frame_id']: record for record in records}
    for frame, group in rows.loc[rows.class_name.eq('person')].groupby('frame_id'):
        record = lookup[frame]
        boxes = np.asarray(record['boxes'], float).reshape(-1, 4)
        confidence = np.asarray(record['confidence'])
        keep = confidence >= .3
        boxes, confidence = boxes[keep], confidence[keep]
        for index, row in group.iterrows():
            if not len(boxes):
                continue
            proposal = row[BOX].to_numpy(float)
            height = max(proposal[3]-proposal[1], 1)
            overlap = np.maximum(0, np.minimum(proposal[2], boxes[:, 2])-np.maximum(proposal[0], boxes[:, 0]))
            union = np.maximum(proposal[2], boxes[:, 2])-np.minimum(proposal[0], boxes[:, 0])
            horizontal = overlap/np.maximum(union, 1)
            top = np.abs(boxes[:, 1]-proposal[1])/height
            lower = (boxes[:, 3]-proposal[3])/height
            strength = expit((lower-.5)/.12) * np.exp(-np.square(top/.15)) * np.clip((horizontal-.4)/.4, 0, 1) * np.clip((confidence-.5)/.4, 0, 1)
            result.loc[index, 'person_truncation'] = float(strength.max())
    return result


def download(root):
    path = Path(root)/'.cache/models/bytetrack_x_mot17.pth.tar'
    if path.exists() and sha256(path) == MODEL_SHA256:
        return path
    import gdown
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    gdown.download(MODEL_URL, str(temporary), fuzzy=True)
    if not temporary.exists() or sha256(temporary) != MODEL_SHA256:
        raise ValueError('Full-body checkpoint checksum mismatch; download was not accepted.')
    temporary.replace(path)
    return path


def preprocess(image):
    """Original ByteTrack normalization (different from modern YOLOX)."""
    import cv2
    padded = np.ones((800, 1440, 3))*114.
    ratio = min(800/image.shape[0], 1440/image.shape[1])
    height, width = int(image.shape[0]*ratio), int(image.shape[1]*ratio)
    padded[:height, :width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded = padded[:, :, ::-1]/255.
    padded -= (.485, .456, .406)
    padded /= (.229, .224, .225)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def infer(rows, data, root, cache, device='auto'):
    """Run the public frozen model only on frames containing scored people."""
    import torch
    from ._bytetrack import YOLOX, YOLOPAFPN, YOLOXHead
    from ._bytetrack.boxes import postprocess
    from .vision import image, release
    frames = sorted(rows.loc[rows.class_name.eq('person'), 'frame_id'].unique())
    signature = hashlib.sha256((MODEL_SHA256+'v1'+''.join(frame+sha256(Path(data)/f'{frame}.jpg') for frame in frames)).encode()).hexdigest()
    path = Path(cache)/f'fullbody_{signature}.json'
    records = json.loads(path.read_text()) if path.exists() else []
    seen = {r['frame_id'] for r in records}
    if seen == set(frames):
        return evidence(rows, records)
    checkpoint = download(root)
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(3)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = YOLOX(YOLOPAFPN(depth=1.33, width=1.25), YOLOXHead(1, width=1.25))
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval().requires_grad_(False).to(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, 1):
        if frame in seen:
            continue
        tensor, ratio = preprocess(image(data, frame))
        with torch.inference_mode():
            prediction = model(torch.from_numpy(tensor)[None].to(device)).cpu()
        output = postprocess(prediction.clone(), 1, conf_thre=.01, nms_thre=.7)[0]
        if output is None:
            boxes, confidence = np.empty((0, 4)), np.empty(0)
        else:
            boxes = np.clip(output[:, :4].numpy()/ratio, 0, [1280, 720, 1280, 720])
            confidence = (output[:, 4]*output[:, 5]).numpy()
        records.append(dict(frame_id=frame, boxes=boxes.tolist(), confidence=confidence.tolist()))
        path.write_text(json.dumps(records))
        print(f'Full-body person: {index}/{len(frames)}', flush=True)
    del model
    release(device)
    return evidence(rows, records)
