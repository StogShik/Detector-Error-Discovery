"""Raw-image inference for exactly the six selected visual models."""
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

from .core import BOX, NAMES, SMALL, nms, risks, sha256

WINDOWS = [(x, y, x+512, y+384) for y in [0, 336] for x in [0, 384, 768]]


def image(data, frame):
    value = cv2.imread(str(Path(data)/f'{frame}.jpg'))
    if value is None:
        raise FileNotFoundError(f'Missing image: {frame}.jpg')
    return value if value.shape[:2] == (720, 1280) else cv2.resize(value, (1280, 720))


def release(device):
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()


def local_detector(kind, rows, data, root, device):
    model = YOLO(str(root/'models'/f'{kind}.pt'))
    expected = NAMES if kind == 'multi' else (['person'] if kind == 'person' else SMALL)
    assert list(model.names.values()) == expected
    records = []
    for frame in sorted(rows.frame_id.unique()):
        im = image(data, frame)
        if kind != 'tiles':
            pred = model.predict(im, imgsz=1280, conf=.005, iou=.6, device=device, verbose=False)[0]
            b, c, p = pred.boxes.xyxy.cpu().numpy(), pred.boxes.cls.int().cpu().numpy(), pred.boxes.conf.cpu().numpy()
        else:
            boxes, classes, scores = [], [], []
            for left, top, right, bottom in WINDOWS:
                pred = model.predict(im[top:bottom, left:right], imgsz=768, conf=.01, iou=.6, device=device, verbose=False)[0]
                raw = pred.boxes.xyxy.cpu().numpy()
                valid = np.ones(len(raw), bool)
                if left: valid &= raw[:, 0] > 3
                if top: valid &= raw[:, 1] > 3
                if right < 1280: valid &= raw[:, 2] < right-left-3
                if bottom < 720: valid &= raw[:, 3] < bottom-top-3
                boxes.append(raw[valid]+[left, top, left, top])
                classes.append(pred.boxes.cls.int().cpu().numpy()[valid])
                scores.append(pred.boxes.conf.cpu().numpy()[valid])
            b, c, p = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
            keep = nms(b, c, p)
            b, c, p = b[keep], c[keep], p[keep]
        records.append(dict(frame_id=frame, boxes=b.tolist(), classes=c.tolist(), confidence=p.tolist()))
    del model
    release(device)
    return records


def clothing(rows, data, root, cache, device):
    folder = cache/'clothing_crops'
    folder.mkdir(exist_ok=True)
    for row in rows.itertuples():
        with Image.open(Path(data)/f'{row.frame_id}.jpg') as source:
            im = source.convert('RGB')
            if im.size != (1280, 720): im = im.resize((1280, 720))
            box = (max(0, int(np.floor(row.x1))), max(0, int(np.floor(row.y1))), min(1280, int(np.ceil(row.x2))), min(720, int(np.ceil(row.y2))))
            ImageOps.pad(im.crop(box), (224, 224), method=Image.Resampling.BILINEAR, color=(124, 116, 104)).save(folder/f'{row.id}.png')
    model = YOLO(str(root/'models/clothing.pt'))
    index = next(k for k, v in model.names.items() if v == 'clothes')
    values = []
    for start in range(0, len(rows), 16):
        paths = [str(folder/f'{i}.png') for i in rows.iloc[start:start+16].id]
        values.extend(float(r.probs.data[index].cpu()) for r in model.predict(paths, imgsz=224, device=device, verbose=False))
    del model
    release(device)
    return pd.DataFrame(dict(id=rows.id, p_clothes=values))


class Grounded:
    def __init__(self, config, device):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor = AutoProcessor.from_pretrained(config['grounding_model'], revision=config['grounding_revision'])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(config['grounding_model'], revision=config['grounding_revision']).eval().requires_grad_(False)
        self.model.config.disable_custom_kernels = True
        self.model.to(device)
        self.device = device

    def predict(self, im, small=False):
        words = ['hand', 'cup', 'glass', 'tray', 'glove', 'bottle'] if small else ['person', 'hand', 'cup', 'glass', 'tray']
        mapping = np.array([1, 2, 2, 3, 4, 5] if small else [0, 1, 2, 2, 3])
        prompt = '. '.join(words)+'.'
        tokenized = self.processor.tokenizer(prompt, return_offsets_mapping=True)
        positions, cursor = [], 0
        for word in words:
            start = prompt.index(word, cursor); stop = start+len(word); cursor = stop
            positions.append([i for i, (a, b) in enumerate(tokenized['offset_mapping']) if b > a and a >= start and b <= stop])
        options = dict(size={'shortest_edge': 640, 'longest_edge': 1024}) if small else {}
        inputs = self.processor(images=cv2.cvtColor(im, cv2.COLOR_BGR2RGB), text=prompt, return_tensors='pt', **options).to(self.device)
        with torch.inference_mode(): output = self.model(**inputs)
        logits = output.logits[0].sigmoid().cpu().numpy()
        scores = np.column_stack([logits[:, p].mean(axis=1) for p in positions])
        cls, conf = scores.argmax(axis=1), scores.max(axis=1)
        valid = conf >= (.08 if small else .06)
        cls, conf = mapping[cls[valid]], conf[valid]
        xywh = output.pred_boxes[0].cpu().numpy()[valid]
        h, w = im.shape[:2]
        b = np.column_stack([xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2])*[w, h, w, h]
        b = np.clip(b, 0, [w, h, w, h]); keep = nms(b, cls, conf)
        return b[keep], cls[keep], conf[keep]


def grounding(rows, data, config, cache, device):
    whole_path, crops_path = cache/'grounded_whole.json', cache/'grounded_crops.json'
    whole = json.loads(whole_path.read_text()) if whole_path.exists() else []
    crops = json.loads(crops_path.read_text()) if crops_path.exists() else []
    seen_frames, seen_ids = {r['frame_id'] for r in whole}, {r['id'] for r in crops}
    net = Grounded(config, device)
    frames = sorted(rows.loc[rows.class_name.eq('person'), 'frame_id'].unique())
    for i, frame in enumerate(frames, 1):
        if frame in seen_frames: continue
        b, c, p = net.predict(image(data, frame))
        whole.append(dict(frame_id=frame, boxes=b.tolist(), classes=c.tolist(), confidence=p.tolist()))
        whole_path.write_text(json.dumps(whole))
        if i % 10 == 0: print(f'Grounding DINO whole: {i}/{len(frames)}', flush=True)
    small = rows.loc[rows.class_name.isin(SMALL)]
    for i, row in enumerate(small.itertuples(), 1):
        if row.id in seen_ids: continue
        a = np.array([row.x1, row.y1, row.x2, row.y2]); center = (a[:2]+a[2:])/2
        half = np.maximum((a[2:]-a[:2])*3, 48)
        bounds = np.clip(np.r_[np.floor(center-half), np.ceil(center+half)].astype(int), 0, [1280, 720, 1280, 720])
        x1, y1, x2, y2 = bounds
        b, c, p = net.predict(image(data, row.frame_id)[y1:y2, x1:x2], small=True)
        boundary = np.zeros(len(b), bool)
        if x1: boundary |= b[:, 0] < 2
        if y1: boundary |= b[:, 1] < 2
        if x2 < 1280: boundary |= b[:, 2] > x2-x1-2
        if y2 < 720: boundary |= b[:, 3] > y2-y1-2
        use = np.isin(c, [1, 2, 3]) & ~boundary
        crops.append(dict(id=row.id, frame_id=row.frame_id, boxes=(b[use]+[x1, y1, x1, y1]).tolist(), classes=c[use].tolist(), confidence=p[use].tolist()))
        crops_path.write_text(json.dumps(crops))
        if i % 10 == 0: print(f'Grounding DINO crops: {i}/{len(small)}', flush=True)
    merged = []
    for frame in sorted({r['frame_id'] for r in crops}):
        selected = [r for r in crops if r['frame_id'] == frame]
        b = np.asarray([b for r in selected for b in r['boxes']], float).reshape(-1, 4)
        c = np.asarray([c for r in selected for c in r['classes']], int)
        p = np.asarray([p for r in selected for p in r['confidence']])
        keep = nms(b, c, p)
        merged.append(dict(frame_id=frame, boxes=b[keep].tolist(), classes=c[keep].tolist(), confidence=p[keep].tolist()))
    del net
    release(device)
    return whole, merged


def infer(features, data, root, config, cache, device='auto'):
    from .hands import extract, summarize
    import hashlib
    data, root, cache = Path(data), Path(root), Path(cache)
    if device == 'auto': device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    image_hashes = ''.join(sha256(data/f'{f}.jpg') for f in sorted(features.frame_id.unique()))
    signature = hashlib.sha256((features.to_csv(index=False)+json.dumps(config, sort_keys=True)+image_hashes).encode()).hexdigest()
    cache = cache/signature[:16]; cache.mkdir(parents=True, exist_ok=True)
    person = features.class_name.eq('person'); small = features.class_name.isin(SMALL)
    cp = cache/'clothing.csv'
    cloth = pd.read_csv(cp, dtype={'id': str}) if cp.exists() else clothing(features.loc[person], data, root, cache, device)
    cloth.to_csv(cp, index=False)
    features = features.copy(); features['p_clothes'] = features.id.map(cloth.set_index('id').p_clothes)
    raw = {}
    for kind, subset in [('multi', features), ('person', features.loc[person]), ('tiles', features.loc[small])]:
        path = cache/f'{kind}.json'
        raw[kind] = json.loads(path.read_text()) if path.exists() else local_detector(kind, subset, data, root, device)
        path.write_text(json.dumps(raw[kind])); print(f'Finished {kind} detector', flush=True)
    teacher = risks(features, raw['multi'], legacy=True)
    teacher.loc[person] = risks(features.loc[person], raw['person'], ['person'], legacy=True)
    tile = risks(features.loc[small], raw['tiles'], SMALL, legacy=True)
    teacher.loc[small] = .5*teacher.loc[small]+.5*tile
    whole, cropped = grounding(features, data, config, cache, device)
    signals = features[['id']].copy(); signals['teacher_risk'] = teacher
    signals['grounded_person'] = risks(features.loc[person], whole, threshold=.3)
    signals['grounded_small'] = risks(features.loc[small], cropped, threshold=.2)
    hands = summarize(features.loc[features.class_name.eq('nude_hands')], extract(features, data, root/'models/hand_landmarker.task', cache/'hands.json')).set_index('id')
    signals['hand_found'] = features.id.map(hands.hand_found).fillna(False).astype(bool)
    signals['hand_risk'] = features.id.map(hands.risk)
    signals['hand_consensus'] = features.id.map(hands.consensus).fillna(0)
    features.to_csv(cache/'features.csv', index=False); signals.to_csv(cache/'signals.csv', index=False)
    return features, signals
