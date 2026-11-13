"""Audit and prepare manually reviewed TRAIN person annotations safely."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT/'training/annotations/diverse_person'
BOX = ['x1', 'y1', 'x2', 'y2']
VISIBLE = ['visible_x1', 'visible_y1', 'visible_x2', 'visible_y2']


def seconds(frame):
    time = frame[-6:]
    return int(time[:2])*3600+int(time[2:4])*60+int(time[4:])


def intersection(a, b):
    return np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()


def audit(data=None):
    rows = pd.read_csv(ANNOTATIONS/'observations.csv', dtype={'day': str})
    geometry = pd.read_csv(ANNOTATIONS/'geometry_boxes.csv', dtype={'day': str})
    frames = pd.read_csv(ANNOTATIONS/'frames.csv', dtype={'day': str})
    policy = json.loads((ANNOTATIONS/'policy.json').read_text())
    assert rows.annotation_id.is_unique and frames.frame_id.is_unique
    assert len(rows) == policy['observations'] and len(frames) == policy['frames']
    assert set(rows.frame_id) <= set(frames.frame_id)
    assert set(rows.day) <= {'20261025', '20261026'}
    np.testing.assert_array_equal(rows.day.astype(str), rows.frame_id.str.rsplit('_', n=2).str[-2].astype(str))
    visible = rows[VISIBLE].to_numpy(float)
    assert np.isfinite(visible).all() and (visible[:, 2:] > visible[:, :2]).all()
    assert (visible >= 0).all() and (visible <= [1280, 720, 1280, 720]).all()
    unsupported = ~rows.full_extent_observable
    assert rows.loc[unsupported, BOX].isna().all().all()
    assert not rows.loc[unsupported, 'geometry_usable'].any()
    assert not rows.loc[rows.pose_redundant, 'geometry_usable'].any()
    assert not rows.extent_inferred.any()
    np.testing.assert_array_equal(rows.loc[rows.full_extent_observable, BOX], rows.loc[rows.full_extent_observable, VISIBLE])
    np.testing.assert_allclose(rows.loc[rows.full_extent_observable, ['native_x1', 'native_y1', 'native_x2', 'native_y2']]*2,
                               rows.loc[rows.full_extent_observable, BOX], atol=1e-12, rtol=0)
    assert set(geometry.annotation_id) == set(rows.loc[rows.geometry_usable, 'annotation_id'])
    np.testing.assert_allclose(geometry.set_index('annotation_id')[BOX].sort_index(),
                               rows.loc[rows.geometry_usable].set_index('annotation_id')[BOX].sort_index(),
                               atol=1e-12, rtol=0)
    coco = json.loads((ANNOTATIONS/'visible_people.coco.json').read_text())
    assert len(coco['annotations']) == len(rows)
    by_id = rows.set_index('annotation_id')
    for annotation in coco['annotations']:
        row = by_id.loc[annotation['attributes']['annotation_id']]
        box = row[VISIBLE].to_numpy(float)/2
        np.testing.assert_allclose(annotation['bbox'], [box[0], box[1], box[2]-box[0], box[3]-box[1]], atol=1e-12, rtol=0)
        if not row.full_extent_observable:
            assert annotation['attributes']['full_extent_box_xyxy_native'] is None
    previous = pd.read_csv(ANNOTATIONS/'previously_annotated_frames.csv', dtype={'day': str})
    assert not set(frames.frame_id) & set(previous.frame_id)
    gaps = []
    for day, group in frames.groupby('day'):
        times = np.sort(group.frame_id.map(seconds).to_numpy())
        assert (np.diff(times) >= 300).all()
        before = previous.loc[previous.day.eq(day), 'frame_id'].map(seconds).to_numpy()
        gaps.extend(np.abs(times[:, None]-before[None]).min(axis=1))
    assert min(gaps) >= 180
    if data is not None:
        data = Path(data)
        test = set(pd.read_csv(data/'test_frames.csv').frame_id)
        allowed = set(pd.read_csv(data/'machine_frames.csv').frame_id)
        original = set(pd.read_csv(data/'human_boxes_train.csv').frame_id)
        assert not set(frames.frame_id) & (test | original)
        assert set(frames.frame_id) <= allowed
        hashes = json.loads((ANNOTATIONS/'image_sha256.json').read_text())
        for row in frames.itertuples():
            path = data/row.image_path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[row.frame_id]
            assert cv2.imread(str(path)).shape[:2] == (360, 640)
    report = dict(reviewed_frames=len(frames), reviewed_people=len(rows),
                  supported_full_boxes=int(rows.full_extent_observable.sum()),
                  nonredundant_geometry_boxes=len(geometry),
                  hidden_full_targets_withheld=int(unsupported.sum()),
                  minimum_prior_frame_gap_seconds=int(min(gaps)), raw_images_checked=data is not None,
                  source='Assistant visual review with model hints; not organizer GT.',
                  poses=rows.pose.value_counts().to_dict())
    return rows, frames, report


def build(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    rows, frames, report = audit(data)
    folder = output/'dataset'
    for directory in ['images', 'labels']:
        (folder/directory).mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(data/'human_boxes_train.csv')
    records, rejected = [], []
    for frame in sorted(human.frame_id.unique()):
        destination = folder/'images'/f'{frame}.jpg'
        if not destination.exists():
            destination.symlink_to(data/f'{frame}.jpg')
        group = human.loc[human.frame_id.eq(frame) & human.class_name.eq('person')]
        (folder/'labels'/f'{frame}.txt').write_text('\n'.join(f'0 {r.cx} {r.cy} {r.w} {r.h}' for r in group.itertuples()))
        records.append(dict(frame_id=frame, image=str(destination), day=frame.rsplit('_', 2)[-2], source='original', boxes=len(group)))
    for frame, group in rows.groupby('frame_id'):
        if not group.geometry_usable.any():
            continue
        known, unknown = group.loc[group.full_extent_observable], group.loc[~group.full_extent_observable]
        masks = []
        bad = False
        for _, row in unknown.iterrows():
            mask = np.clip(row[VISIBLE].to_numpy(float)/2+[-3, -3, 3, 3], 0, [640, 360, 640, 360])
            for _, target in known.iterrows():
                box = target[BOX].to_numpy(float)/2
                if intersection(mask, box)/max((box[2:]-box[:2]).prod(), 1) > .03:
                    bad = True
            masks.append(mask)
        if bad:
            rejected.append(dict(frame_id=frame, reason='Unknown-person mask would remove >3% of a supported target.'))
            continue
        row = group.iloc[0]
        name = f'context_D{int(row.frame_index):03d}'
        image = cv2.imread(str(data/row.image_path))
        for mask in masks:
            a, b, c, d = np.rint(mask).astype(int)
            image[b:d, a:c] = 114
        path = folder/'images'/f'{name}.jpg'
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels = []
        for _, target in known.iterrows():
            a, b, c, d = target[BOX].to_numpy(float)/2
            labels.append(f'0 {(a+c)/1280} {(b+d)/720} {(c-a)/640} {(d-b)/360}')
        (folder/'labels'/f'{name}.txt').write_text('\n'.join(labels))
        records.append(dict(frame_id=frame, image=str(path), day=row.day, source='new_masked_context',
                            boxes=len(labels), ignore_boxes=[m.tolist() for m in masks]))
    inventory = pd.DataFrame(records)
    inventory.to_csv(output/'training_manifest.csv', index=False)
    pd.DataFrame(rejected).to_csv(output/'context_rejections.csv', index=False)
    report['training_images_by_day_source'] = {f'{day}:{source}': int(len(group)) for (day, source), group in inventory.groupby(['day', 'source'])}
    report['training_boxes_by_day_source'] = {f'{day}:{source}': int(group.boxes.sum()) for (day, source), group in inventory.groupby(['day', 'source'])}
    report['rejected_context_frames'] = len(rejected)
    (output/'annotation_audit.json').write_text(json.dumps(report, indent=2))
    return inventory, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diverse_person')
    args = parser.parse_args()
    if args.prepare:
        if args.data_dir is None:
            parser.error('--prepare requires --data-dir')
        _, result = build(args.data_dir, args.output)
    else:
        _, _, result = audit(args.data_dir)
    print(json.dumps(result, indent=2))
