"""Controlled same-day person-teacher continuation on reviewed extra TRAIN data."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from ultralytics import YOLO

from detector_error_discovery.core import BOX, risks
from diverse_person_data import build, audit

ROOT = Path(__file__).resolve().parents[1]
DAYS = ['20261025', '20261026']
CONFIG = dict(epochs=6, imgsz=640, batch=4, freeze=10, lr0=.0001, seed=373,
              mosaic=0., degrees=3, scale=.15, original_oversample=2,
              unknown_full_extents_are_targets=False, step_matched_control=True, inference_imgsz=960)


def graph(rows, person_risk=None):
    """Recompose the full reference graph while changing its person teacher."""
    person = rows.class_name.eq('person')
    p = rows.p_clothes.fillna(0)
    teacher = rows.teacher_risk.copy()
    if person_risk is not None:
        teacher.loc[person] = person_risk
    value = .6*rows.selected+.4*teacher+.4*p*(1-teacher)
    value.loc[person] = .85*value.loc[person]+.15*(rows.loc[person, 'roi_risk']+p.loc[person]*(1-rows.loc[person, 'roi_risk']))
    mixed = teacher.copy()
    mixed.loc[person] = .5*teacher.loc[person]+.5*rows.loc[person, 'active_risk']
    order = .6*rows.selected+.4*mixed+.4*p*(1-mixed)
    order.loc[person] = .85*order.loc[person]+.15*(rows.loc[person, 'roi_risk']+p.loc[person]*(1-rows.loc[person, 'roi_risk']))
    for _, group in rows.loc[person].groupby('day'):
        index = order.loc[group.index].sort_values(kind='stable').index
        value.loc[index] = np.sort(value.loc[group.index])
    value.loc[person] = .5*value.loc[person]+.5*(rows.loc[person, 'grounded_person']+p.loc[person]*(1-rows.loc[person, 'grounded_person']))
    for cls, weight in [('nude_hands', .25), ('cup', .5)]:
        mask = rows.class_name.eq(cls)
        value.loc[mask] = (1-weight)*value.loc[mask]+weight*rows.loc[mask, 'grounded_small']
    for _, group in rows.loc[rows.class_name.eq('tray')].groupby('day'):
        order = (.75*value.loc[group.index]+.25*rows.loc[group.index, 'grounded_small']).sort_values(kind='stable').index
        value.loc[order] = np.sort(value.loc[group.index])
    found = rows.class_name.eq('nude_hands') & rows.hand_found
    value.loc[found] = .7*value.loc[found]+.3*rows.loc[found, 'hand_risk']
    return value


def evaluate(output):
    rows = pd.read_csv(ROOT/'reference/diverse_person_graph_inputs.csv', dtype={'id': str, 'day': str})
    baseline = graph(rows)
    np.testing.assert_allclose(baseline, rows.baseline, atol=1e-12, rtol=0)
    person = rows.class_name.eq('person')
    p = rows.p_clothes.fillna(0)
    scores = {'reference_public_087853': baseline}
    for kind in ['control', 'augmented']:
        records = []
        for day in DAYS:
            path = output/f'{kind}_{day}_oof.json'
            if not path.exists():
                path = ROOT/'reference'/f'diverse_person_{kind}_{day}_oof.json'
            records += json.loads(path.read_text())
        risk = risks(rows.loc[person], records, names=['person'], legacy=True)
        protected = risk+p.loc[person]*(1-risk)
        scores[kind+'_replacement'] = graph(rows, risk)
        for weight in [.25, .5]:
            s = baseline.copy()
            s.loc[person] = (1-weight)*s.loc[person]+weight*protected
            name = f'{kind}_w{weight}'
            scores[name] = s
            rerank = baseline.copy()
            for _, group in rows.loc[person].groupby('day'):
                order = s.loc[group.index].sort_values(kind='stable').index
                rerank.loc[order] = np.sort(baseline.loc[group.index])
            scores[name+'_rerank'] = rerank
    report = []
    for day, group in rows.groupby('day'):
        for subset in ['all', 'person', 'test_like']:
            index = group.index if subset == 'all' else group.index[group.class_name.eq('person')] if subset == 'person' else group.index[group.class_name.ne('person') | group.confidence.ge(.73)]
            for name, s in scores.items():
                report.append(dict(day=day, subset=subset, method=name, AP=average_precision_score(rows.loc[index, 'error'], s.loc[index])))
    report = pd.DataFrame(report)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output/'validation.csv', index=False)
    pd.DataFrame(scores).assign(id=rows.id).to_csv(output/'oof_scores.csv', index=False)
    tables = {}
    for subset in report.subset.unique():
        table = report.loc[report.subset.eq(subset)].pivot(index='method', columns='day', values='AP')
        table['mean'] = table.mean(axis=1)
        tables[subset] = table
        table.to_csv(output/f'{subset}_summary.csv')
        print(subset+'\n'+table.sort_values('mean', ascending=False).to_string())
    a, p = tables['all'], tables['person']
    reference, eligible = 'reference_public_087853', []
    for name in a.index:
        if not name.startswith('augmented'):
            continue
        control = name.replace('augmented', 'control')
        if ((a.loc[name, DAYS] >= a.loc[reference, DAYS]).all() and
                (p.loc[name, DAYS] >= p.loc[reference, DAYS]).all() and
                a.loc[name, 'mean'] >= a.loc[reference, 'mean']+.002 and
                p.loc[name, 'mean'] >= p.loc[reference, 'mean']+.01 and
                a.loc[name, 'mean'] >= a.loc[control, 'mean']+.001 and
                (tables['test_like'].loc[name, DAYS] >= tables['test_like'].loc[reference, DAYS]).all()):
            eligible.append(name)
    selected = a.loc[eligible, 'mean'].idxmax() if eligible else None
    result = dict(chosen=selected, public_AP=None, default_changed=False,
                  criterion='Both days overall/person/test-like nondecreasing; mean overall gain >=.002, person >=.01; beats the step-matched no-new-data control by .001 overall.',
                  caveat='Exploratory comparison on two repeatedly reused days; manual extra annotations are estimates, not organizer GT.')
    (output/'selection.json').write_text(json.dumps(result, indent=2))
    print('SELECTED', selected)
    return result


def warm_start(data, output, day, device, directory):
    if directory is not None:
        path = directory/f'person_holdout_{day}.pt'
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    # Rebuild the exact two-stage warm-start recipe using bundled TRAIN labels.
    from train import detector_data, fit_detector
    work = output/f'warm_{day}'
    work.mkdir(parents=True, exist_ok=True)
    public = work/'yolo11s.pt'
    if not public.exists():
        urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt', public)
    folder, inventory, names = detector_data(data, work, 'original_person', day)
    initial = fit_detector(public, folder, inventory, names, work, 'person_original_warm', device, 20, 640, 0, .001)
    folder, inventory, names = detector_data(data, work, 'person', day)
    return fit_detector(initial, folder, inventory, names, work, 'person_annotation_warm', device, 12, 640, 17, .0003, repeat=True)


def train(inventory, kind, day, initial, output, device):
    full = inventory.loc[inventory.day.ne(day)]
    fit = full if kind == 'augmented' else full.loc[full.source.eq('original')]
    paths = fit.image.tolist()+fit.loc[fit.source.eq('original') & fit.boxes.gt(0), 'image'].tolist()
    count = len(full)+int((full.source.eq('original') & full.boxes.gt(0)).sum())
    paths += [paths[i % len(paths)] for i in range(max(0, count-len(paths)))]
    assert not fit.day.eq(day).any() and len(paths) == count
    labels_hash = ''.join(hashlib.sha256((Path(p).parent.parent/'labels'/f'{Path(p).stem}.txt').read_bytes()).hexdigest() for p in paths)
    config = dict(CONFIG, kind=kind, holdout=day, frame_ids=fit.frame_id.tolist(), exposures=len(paths),
                  warm_start_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(), labels_sha256=hashlib.sha256(labels_hash.encode()).hexdigest())
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    checkpoint = output/f'{kind}_{day}.pt'
    meta = checkpoint.with_suffix('.json')
    if checkpoint.exists():
        assert json.loads(meta.read_text())['signature'] == signature
        return checkpoint
    listing = output/f'{kind}_{day}.txt'
    listing.write_text('\n'.join(paths))
    data = output/f'{kind}_{day}.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(output/'dataset'), train=str(listing), val=str(listing), names={0: 'person'})))
    name = f'run_{kind}_{day}'
    model = YOLO(str(initial))
    model.train(data=str(data), epochs=6, imgsz=640, batch=4, workers=0, device=device,
                freeze=10, optimizer='AdamW', lr0=.0001, lrf=.1, weight_decay=.0005,
                mosaic=0., degrees=3, scale=.15, fliplr=.5, seed=373, deterministic=True,
                patience=6, val=False, plots=False, amp=False, project=str(output), name=name, exist_ok=True)
    shutil.copy2(output/name/'weights/last.pt', checkpoint)
    meta.write_text(json.dumps(dict(signature=signature, **config), indent=2))
    del model
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    return checkpoint


def verify(checkpoint, rows, data, output, kind, day, device):
    path = output/f'{kind}_{day}_oof.json'
    if path.exists():
        return
    net = YOLO(str(checkpoint))
    records = []
    for frame in sorted(rows.loc[rows.day.eq(day) & rows.class_name.eq('person'), 'frame_id'].unique()):
        prediction = net.predict(str(data/f'{frame}.jpg'), imgsz=960, conf=.005, iou=.6, device=device, verbose=False)[0]
        records.append(dict(frame_id=frame, boxes=prediction.boxes.xyxy.cpu().numpy().tolist(),
                            classes=prediction.boxes.cls.int().cpu().numpy().tolist(), confidence=prediction.boxes.conf.cpu().numpy().tolist()))
    path.write_text(json.dumps(records))
    del net
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'dataset', 'cv', 'evaluate'], default='evaluate')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diverse_person_cv')
    parser.add_argument('--warm-start-dir', type=Path, help='Optional trusted same-fold person_holdout_DAY.pt files')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.stage == 'audit':
        _, _, result = audit(args.data_dir)
        print(json.dumps(result, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'evaluate':
        evaluate(args.output)
        return
    if args.data_dir is None:
        parser.error('dataset/cv requires --data-dir')
    inventory, report = build(args.data_dir, args.output)
    print(json.dumps(report, indent=2))
    if args.stage == 'dataset':
        return
    device = args.device if args.device != 'auto' else 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(3)
    rows = pd.read_csv(ROOT/'reference/diverse_person_graph_inputs.csv', dtype={'id': str, 'day': str})
    for day in DAYS:
        initial = warm_start(args.data_dir, args.output, day, device, args.warm_start_dir)
        for kind in ['control', 'augmented']:
            checkpoint = train(inventory, kind, day, initial, args.output, device)
            verify(checkpoint, rows, args.data_dir, args.output, kind, day, device)
    evaluate(args.output)


if __name__ == '__main__':
    main()
