"""Rebuild the selected weights from provided TRAIN data and reviewed labels."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageOps
from catboost import CatBoostClassifier
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

from detector_error_discovery.core import NAMES, SMALL, sha256
from detector_error_discovery.features import build

ROOT = Path(__file__).resolve().parents[1]


class WholeCropTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = ClassificationDataset(img_path, self.args, augment=mode=='train', prefix=mode, names=self.data['names'])
        transforms = [T.Resize((224, 224))]
        if mode == 'train': transforms += [T.RandomHorizontalFlip(.5), T.ColorJitter(.15, .15, .1, .01)]
        dataset.torch_transforms = T.Compose(transforms+[T.ToTensor()])
        return dataset


def day(frame): return frame.rsplit('_', 2)[-2]


def detector_data(data, work, kind, holdout):
    human = pd.read_csv(data/'human_boxes_train.csv')
    human['class_name'] = human.class_name.replace({'nude_hand': 'nude_hands'})
    frames = sorted(human.frame_id.unique())
    names = NAMES
    if kind == 'person':
        human = pd.read_csv(ROOT/'training/annotations/person_boxes.csv')
        frames = pd.read_csv(data/'train_frames.csv').frame_id.tolist()
        names = ['person']
    if kind == 'original_person':
        human = human.loc[human.class_name.eq('person')]; names = ['person']
    if kind == 'tiles': names = SMALL
    human = human.loc[human.class_name.isin(names)]
    folder = work/f'dataset_{kind}'
    for directory in ['images', 'labels']: (folder/directory).mkdir(parents=True, exist_ok=True)
    originals = set(pd.read_csv(data/'human_boxes_train.csv').query('class_name == "person"').frame_id)
    manifest = []
    windows = [(x, y, x+512, y+384) for y in [0, 336] for x in [0, 384, 768]]
    for frame in frames:
        if day(frame) == holdout: continue
        group = human.loc[human.frame_id.eq(frame)]
        if kind != 'tiles':
            image = folder/'images'/f'{frame}.jpg'
            if not image.exists(): image.symlink_to((data/f'{frame}.jpg').resolve())
            lines = []
            for r in group.itertuples():
                x1, y1 = max(0, r.cx-r.w/2), max(0, r.cy-r.h/2)
                x2, y2 = min(1, r.cx+r.w/2), min(1, r.cy+r.h/2)
                lines.append(f'{names.index(r.class_name)} {(x1+x2)/2} {(y1+y2)/2} {x2-x1} {y2-y1}')
            (folder/'labels'/f'{frame}.txt').write_text('\n'.join(lines))
            manifest.append(dict(frame_id=frame, image=str(image), boxes=len(lines), original=frame in originals))
        else:
            im = cv2.imread(str(data/f'{frame}.jpg')); candidates = []
            for tile, (left, top, right, bottom) in enumerate(windows):
                lines = []
                for r in group.itertuples():
                    a, b, c, d = (r.cx-r.w/2)*1280, (r.cy-r.h/2)*720, (r.cx+r.w/2)*1280, (r.cy+r.h/2)*720
                    x1, y1, x2, y2 = max(left, a), max(top, b), min(right, c), min(bottom, d)
                    if max(0, x2-x1)*max(0, y2-y1)/max(1, (c-a)*(d-b)) < .5: continue
                    lines.append(f'{names.index(r.class_name)} {(x1+x2-2*left)/1024} {(y1+y2-2*top)/768} {(x2-x1)/512} {(y2-y1)/384}')
                candidates.append((tile, lines))
            allowed = {t for t, lines in candidates if not lines}
            allowed = set(sorted(allowed)[:1 if any(lines for _, lines in candidates) else 2])
            for tile, lines in candidates:
                if not lines and tile not in allowed: continue
                left, top, right, bottom = windows[tile]; name = f'{frame}_tile{tile}'
                image = folder/'images'/f'{name}.jpg'; cv2.imwrite(str(image), im[top:bottom, left:right], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (folder/'labels'/f'{name}.txt').write_text('\n'.join(lines))
                manifest.append(dict(frame_id=frame, image=str(image), boxes=len(lines), original=False))
    result = pd.DataFrame(manifest); result.to_csv(folder/'manifest.csv', index=False)
    return folder, result, names


def fit_detector(initial, folder, inventory, names, work, name, device, epochs, size, seed, lr, *, repeat=False, **kwargs):
    paths = inventory.image.tolist()
    if repeat: paths += [r.image for r in inventory.itertuples() if r.original for _ in range(2)]
    listing = folder/f'{name}.txt'; listing.write_text('\n'.join(paths))
    config = folder/f'{name}.yaml'; config.write_text(yaml.safe_dump(dict(path=str(folder), train=str(listing), val=str(listing), names=dict(enumerate(names)))))
    checkpoint = work/f'{name}.pt'
    if checkpoint.exists(): return checkpoint
    model = YOLO(str(initial))
    recipes = json.loads((ROOT/'training/detector_recipes.json').read_text())
    params = dict(recipes[name]); params.update(kwargs)
    model.train(data=str(config), device=device, project=str(work/'runs'), name=name, exist_ok=True, **params)
    shutil.copy2(work/'runs'/name/'weights/last.pt', checkpoint)
    return checkpoint


def train_detectors(data, out, work, device, holdout, selected):
    public = work/'yolo11s.pt'
    if not public.exists(): urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt', public)
    warm_day = sorted({day(f) for f in pd.read_csv(data/'human_boxes_train.csv').frame_id if day(f) != holdout})[-1]
    excluded = '20261025' if warm_day == '20261026' else '20261026'
    for kind in ['multi', 'person', 'tiles']:
        if selected not in ['all', kind]: continue
        folder, inventory, names = detector_data(data, work, kind, holdout)
        if kind == 'tiles':
            # Match the selected tiled teacher's training settings.
            checkpoint = fit_detector(public, folder, inventory, names, work, 'tiles_final', device, 18, 768, 41, .001)
        else:
            warm = inventory.loc[inventory.frame_id.map(day).eq(warm_day)]
            initial = public
            if kind == 'person':
                f, inv, n = detector_data(data, work, 'original_person', excluded)
                initial = fit_detector(public, f, inv, n, work, 'person_original_warm', device, 20, 640, 0, .001)
            initial = fit_detector(initial, folder, warm, names, work, kind+'_annotation_warm', device,
                12 if kind == 'person' else 30, 640 if kind == 'person' else 960, 17, .0003 if kind == 'person' else .001, repeat=kind=='person')
            checkpoint = fit_detector(initial, folder, inventory, names, work, kind+'_final', device,
                8 if kind=='person' else 12, 640 if kind=='person' else 960, 41, .0003, repeat=kind=='person')
        shutil.copy2(checkpoint, out/'models'/f'{kind}.pt')


def train_clothing(data, out, work, device, holdout):
    labels = pd.read_csv(ROOT/'training/annotations/semantic_labels.csv')
    labels = labels.loc[labels.semantic_label.isin(['person', 'clothes']) & labels.frame_id.map(day).ne(holdout)]
    folder = work/'dataset_clothing'; counts = labels.semantic_label.value_counts()
    for split in ['train', 'val']:
        for cls in ['clothes', 'person']: (folder/split/cls).mkdir(parents=True, exist_ok=True)
    for r in labels.itertuples():
        with Image.open(data/f'{r.frame_id}.jpg') as im:
            w, h = im.size; box = (max(0, int(np.floor(r.x1))), max(0, int(np.floor(r.y1))), min(w, int(np.ceil(r.x2))), min(h, int(np.ceil(r.y2))))
            patch = ImageOps.pad(im.convert('RGB').crop(box), (224, 224), method=Image.Resampling.BILINEAR, color=(124, 116, 104))
        for split in ['train', 'val']:
            repeat = int(np.ceil(counts.max()/counts[r.semantic_label])) if split=='train' else 1
            for i in range(repeat): patch.save(folder/split/r.semantic_label/f'{r.id}_{i}.png')
    public = work/'yolo11s-cls.pt'
    if not public.exists(): urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt', public)
    model = YOLO(str(public))
    model.train(trainer=WholeCropTrainer, data=str(folder), epochs=20, imgsz=224, batch=8, device=device, workers=0,
        freeze=0, optimizer='AdamW', lr0=.0001, lrf=.1, weight_decay=.001, dropout=.2, seed=0,
        patience=20, val=False, plots=False, amp=False, auto_augment=None, erasing=0,
        project=str(work/'runs'), name='clothing', exist_ok=True)
    shutil.copy2(work/'runs/clothing/weights/last.pt', out/'models/clothing.pt')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-dir', type=Path, required=True); p.add_argument('--output', type=Path, default=Path('outputs/retrained'))
    p.add_argument('--kind', choices=['all', 'tabular', 'clothing', 'multi', 'person', 'tiles'], default='all'); p.add_argument('--holdout'); p.add_argument('--device', default='auto'); a = p.parse_args()
    if a.holdout and a.kind != 'all': p.error('--holdout requires --kind all so no full-data component leaks into evaluation')
    data, out = a.data_dir.resolve(), a.output.resolve(); work = out/'work'; work.mkdir(parents=True, exist_ok=True)
    if a.holdout and a.holdout not in {day(f) for f in pd.read_csv(data/'train.csv').frame_id}:
        p.error('Requested holdout is absent from TRAIN')
    plan = dict(holdout=a.holdout, train_sha256=sha256(data/'train.csv'), human_sha256=sha256(data/'human_boxes_train.csv'),
        person_annotations_sha256=sha256(ROOT/'training/annotations/person_boxes.csv'), semantic_annotations_sha256=sha256(ROOT/'training/annotations/semantic_labels.csv'))
    previous = work/'training_plan.json'
    if previous.exists() and json.loads(previous.read_text()) != plan:
        raise ValueError('Training inputs or holdout changed; use a new --output directory.')
    previous.write_text(json.dumps(plan, indent=2))
    shutil.copytree(ROOT/'models', out/'models', dirs_exist_ok=True); config = json.loads((ROOT/'config.json').read_text())
    device = a.device if a.device != 'auto' else 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if a.kind in ['all', 'tabular']:
        rows = build(data, 'train'); rows = rows.loc[rows.day.ne(a.holdout)] if a.holdout else rows
        for i, columns in enumerate(config['catboost_columns']):
            model = CatBoostClassifier(iterations=100 if i==0 else 400, depth=3 if i==0 else 2, learning_rate=.04 if i==0 else .03,
                l2_leaf_reg=10, cat_features=['class_name'], one_hot_max_size=8, random_seed=0, thread_count=2, verbose=False, allow_writing_files=False)
            model.fit(rows[columns], rows.error); model.save_model(str(out/'models'/f'geometry_{i}.cbm'))
    if a.kind in ['all', 'clothing']: train_clothing(data, out, work, device, a.holdout)
    if a.kind in ['all', 'multi', 'person', 'tiles']: train_detectors(data, out, work, device, a.holdout, a.kind)
    for model in config['models']: model['sha256'] = sha256(out/model['file'])
    config['training_holdout'] = a.holdout; (out/'config.json').write_text(json.dumps(config, indent=2)); print('Saved bundle:', out)


if __name__ == '__main__': main()
