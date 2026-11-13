"""Export reviewed native-coordinate person observations to CSV and COCO."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'training/annotations/diverse_person'
SCOPE = 'instance_geometry_only; incomplete full-frame labels must not be trained as exhaustive'


def export(source, output):
    output.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(source/'frames.csv', dtype={'day': str}).set_index('frame_index')
    reviews = json.loads((source/'manual_review.json').read_text())
    assert set(map(int, reviews)) == set(frames.index)
    rows, coco = [], []
    for index in sorted(map(int, reviews)):
        frame = frames.loc[index]
        for number, record in enumerate(reviews[str(index)], 1):
            box = np.asarray(record['visible_box_native'], float)
            assert box.shape == (4,) and np.isfinite(box).all()
            assert (box[2:] > box[:2]).all() and (box >= 0).all() and (box <= [640, 360, 640, 360]).all()
            supported = bool(record['full_extent_observable'])
            duplicate = bool(record.get('pose_redundant', False))
            annotation_id = f'H{index:03d}_{number:02d}'
            note = record['review_note'] or ('Box extrema supported by visible body or image clipping.' if supported else 'Visible extent only: hidden full-body boundary is unknown.')
            row = dict(annotation_id=annotation_id, frame_index=index, frame_id=frame.frame_id,
                       day=frame.day, image_path=frame.image_path, image_width=640, image_height=360,
                       annotation_source='independent_assistant_visual_review', review_status='reviewed',
                       pose=record['pose'], full_extent_observable=supported,
                       geometry_usable=supported and not duplicate, pose_redundant=duplicate,
                       extent_inferred=False, review_note=note, native_tolerance_px=4)
            row.update(dict(zip(['visible_x1', 'visible_y1', 'visible_x2', 'visible_y2'], 2*box)))
            row.update(dict(zip(['x1', 'y1', 'x2', 'y2'], 2*box if supported else [np.nan]*4)))
            row.update(dict(zip(['native_x1', 'native_y1', 'native_x2', 'native_y2'], box if supported else [np.nan]*4)))
            row.update(cx=(box[0]+box[2])/1280 if supported else np.nan,
                       cy=(box[1]+box[3])/720 if supported else np.nan,
                       w=(box[2]-box[0])/640 if supported else np.nan,
                       h=(box[3]-box[1])/360 if supported else np.nan,
                       class_name='person', annotation_scope=SCOPE)
            rows.append(row)
            visible = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            coco.append(dict(id=len(rows), image_id=index, category_id=1, bbox=visible,
                             area=visible[2]*visible[3], iscrowd=0,
                             attributes=dict(annotation_id=annotation_id, pose=record['pose'],
                                             full_extent_observable=supported, geometry_usable=supported and not duplicate,
                                             full_extent_box_xyxy_native=box.tolist() if supported else None,
                                             estimated_hidden_extent=False)))
    table = pd.DataFrame(rows)
    table.to_csv(output/'observations.csv', index=False)
    table.loc[table.geometry_usable].to_csv(output/'geometry_boxes.csv', index=False)
    images = [dict(id=int(i), file_name=f.image_path, width=640, height=360, frame_id=f.frame_id, day=f.day) for i, f in frames.iterrows()]
    package = dict(info=dict(description='TRAIN-only assistant-reviewed visible person extent; hidden full-body targets withheld.',
                            coordinate_system='Native JPEG 640x360; CSV visible/full box coordinates are canonical 1280x720.'),
                   images=images, annotations=coco, categories=[dict(id=1, name='person_visible')])
    (output/'visible_people.coco.json').write_text(json.dumps(package, indent=2))
    print(f'Exported {len(table)} observations; {int(table.geometry_usable.sum())} strict geometry candidates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=SOURCE)
    args = parser.parse_args()
    export(args.source, args.output)
