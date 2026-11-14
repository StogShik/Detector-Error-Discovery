"""Conservative box-level fusion: retain camera-specific hidden-body extent."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from detector_error_discovery.core import BOX,overlaps,risks
from train_diverse_person import graph

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/strong_fusion';OUT.mkdir(exist_ok=True)
OLD=ROOT/'reference/strong_person_old_train.json'


def fuse(old,new,mode):
 lookup={r['frame_id']:r for r in new};records=[]
 for record in old:
  d=lookup[record['frame_id']];a=np.array(record['boxes']).reshape(-1,4);ca=np.array(record['confidence']);b=np.array(d['boxes']).reshape(-1,4);cb=np.array(d['confidence']);active=np.flatnonzero(ca>=.1);keep=cb>=.3;b=b[keep];cb=cb[keep];m=overlaps(a[active],b);boxes=a.copy();conf=ca.copy();used=set()
  if m.size:
   left,right=linear_sum_assignment(-m)
   for local,j in zip(left,right):
    if m[local,j]<.3:continue
    i=active[local];used.add(j);old_h=max(a[i,3]-a[i,1],1);new_h=max(b[j,3]-b[j,1],1)
    if mode=='top_sides':boxes[i,:3]=.5*a[i,:3]+.5*b[j,:3]
    elif .8<=new_h/old_h<=1.25:
     w=.25 if mode=='agree25' else .5;boxes[i]=(1-w)*a[i]+w*b[j]
  added=[]  # Preserve old verifier count and low-confidence matching behavior.
  if added:boxes=np.r_[boxes,b[added]];conf=np.r_[conf,cb[added]]
  records.append(dict(frame_id=record['frame_id'],boxes=boxes.tolist(),confidence=conf.tolist(),classes=np.zeros(len(boxes),int).tolist()))
 return records


def main():
 rows=pd.read_csv(ROOT/'reference/diverse_person_graph_inputs.csv',dtype={'id':str,'day':str});old=json.loads(OLD.read_text())
 sources={'frozen26x':json.loads((ROOT/'reference/strong_person_yolo26x_train.json').read_text())}
 dfine=ROOT/'outputs/dfine_person/dfine_train.json'
 if dfine.exists():sources['dfine']=json.loads(dfine.read_text())
 pose_path=ROOT/'outputs/strong_pose_person/pose_train.json'
 if pose_path.exists():sources['pose26x']=json.loads(pose_path.read_text())
 for name in ['bbox26x','prior50']:
  paths=[ROOT/'outputs/strong_yolo_person'/f'{name}_{d}_oof.json' for d in ['20261025','20261026']]
  if all(p.exists() for p in paths):sources[name]=[r for path in paths for r in json.loads(path.read_text())]
 baseline=graph(rows);scores={'reference_public_087853':baseline};person=rows.class_name.eq('person')
 for name,records in sources.items():
  for mode in ['agree25','agree50','top_sides']:
   raw=fuse(old,records,mode);(OUT/f'{name}_{mode}_train.json').write_text(json.dumps(raw));risk=risks(rows.loc[person],raw,names=['person'],legacy=True);s=graph(rows,risk);label=name+'_'+mode;scores[label]=s;rerank=baseline.copy()
   for _,g in rows.loc[person].groupby('day'):
    order=s.loc[g.index].sort_values(kind='stable').index;rerank.loc[order]=np.sort(baseline.loc[g.index])
   scores[label+'_rerank']=rerank
 report=[]
 for day,g in rows.groupby('day'):
  for subset in ['all','person','test_like']:
   ix=g.index if subset=='all' else g.index[g.class_name.eq('person')] if subset=='person' else g.index[g.class_name.ne('person')|g.confidence.ge(.73)]
   for name,s in scores.items():report.append(dict(day=day,subset=subset,method=name,AP=average_precision_score(rows.loc[ix,'error'],s.loc[ix])))
 report=pd.DataFrame(report);report.to_csv(OUT/'validation.csv',index=False);pd.DataFrame(scores).assign(id=rows.id).to_csv(OUT/'oof_scores.csv',index=False);tables={}
 for subset in report.subset.unique():
  t=report[report.subset.eq(subset)].pivot(index='method',columns='day',values='AP');t['mean']=t.mean(axis=1);tables[subset]=t;t.to_csv(OUT/f'{subset}_summary.csv');print(subset+'\n'+t.sort_values('mean',ascending=False).head(16).to_string())
 a=tables['all'];p=tables['person'];ref='reference_public_087853';days=['20261025','20261026'];valid=(a[days]>=a.loc[ref,days]).all(axis=1)&a['mean'].ge(a.loc[ref,'mean']+.002)&(p[days]>=p.loc[ref,days]).all(axis=1)&p['mean'].ge(p.loc[ref,'mean']+.01)&(tables['test_like'][days]>=tables['test_like'].loc[ref,days]).all(axis=1);selected=a.loc[valid,'mean'].idxmax() if valid.any() else None;(OUT/'selection.json').write_text(json.dumps(dict(chosen=selected,public_AP=None,default_changed=False,caveat='Finite exploratory recipe selection on two reused days; Public improvement unmeasured.'),indent=2));print('SELECTED',selected)
if __name__=='__main__':main()
