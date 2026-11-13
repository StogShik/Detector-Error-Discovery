"""Build only detector-derived features required by the three CatBoost models."""
from pathlib import Path
import numpy as np
from . import scene_features as scene
from . import temporal_features as temporal


def build(data_dir, split):
    root = Path(data_dir)
    rows = scene.build_features(split, root)
    rows = scene.add_graph(rows, split, root)
    rows = temporal.add_features(rows, split, root).reset_index(drop=True)
    rows['aspect_ratio'] = (rows.x2 - rows.x1) / (rows.y2 - rows.y1).clip(lower=1)
    rows['box_area'] = rows.width * rows.height
    rows['bottom'] = rows.y2 / rows.image_height
    rows['top'] = rows.y1 / rows.image_height
    rows['left'] = rows.x1 / rows.image_width
    rows['right'] = rows.x2 / rows.image_width
    rows['edge_margin'] = np.minimum.reduce([rows.left, rows.top, 1-rows.right, 1-rows.bottom])
    rows['log_area'] = np.log(rows.box_area.clip(lower=1e-8))
    rows['log_aspect'] = np.log(rows.aspect_ratio.clip(lower=1e-8))
    return rows
