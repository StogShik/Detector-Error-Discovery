import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

p = argparse.ArgumentParser(); p.add_argument('--predictions', required=True); p.add_argument('--labels', required=True); a = p.parse_args()
pred = pd.read_csv(a.predictions, dtype={'id': str}); labels = pd.read_csv(a.labels, dtype={'id': str})
assert pred.id.is_unique and pred.score.between(0, 1).all() and np.isfinite(pred.score).all()
rows = pred.merge(labels[['id', 'frame_id', 'class_name', 'error']], on='id', validate='one_to_one')
assert len(rows) == len(pred) and rows.class_name.ne('table').all()
rows['day'] = rows.frame_id.str.rsplit('_', n=2).str[-2]
for day, group in rows.groupby('day'):
    print(day, 'detections', len(group), 'AP', average_precision_score(group.error, group.score))
