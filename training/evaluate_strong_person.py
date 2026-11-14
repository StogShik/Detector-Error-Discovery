"""Reproduce selected frozen YOLO26x fusion and its offset-only control."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from build_strong_submission import correction

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=0)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/strong_person_validation.json')
    args=parser.parse_args()
    rows=pd.read_csv(ROOT/'reference/diverse_person_graph_inputs.csv',dtype={'id':str,'day':str})
    old=json.loads((ROOT/'reference/strong_person_old_train.json').read_text())
    new=json.loads((ROOT/'reference/strong_person_yolo26x_train.json').read_text())
    person=rows.class_name.eq('person')
    scores=dict(baseline=rows.baseline,offset_only=(rows.baseline+.05*person)/1.05,
                geometry_only=correction(rows,old,new,offset=0),candidate=correction(rows,old,new))
    metrics=[]
    for day,g in rows.groupby('day'):
        for subset in ['all','person','test_like']:
            ix=g.index if subset=='all' else g.index[g.class_name.eq('person')] if subset=='person' else g.index[g.class_name.ne('person')|g.confidence.ge(.73)]
            for name,score in scores.items():metrics.append(dict(day=day,subset=subset,method=name,AP=average_precision_score(rows.loc[ix,'error'],score.loc[ix])))
    public_result=json.loads((ROOT/'reference/strong_person_public_result.json').read_text())
    result=dict(metrics=metrics,public_AP=public_result['public_AP'],base_public_AP=.87853,default_changed=False,
                selection='Frozen YOLO26x chosen for simplicity: within 0.0002 mean AP of the best explored fine-tuned variant, without new fitted weights.',
                caveat='Finite exploratory recipe selection on two repeatedly reused days; Public measured separately by user leaderboard screenshot; Private unknown. Strict earlier +.01 mean-person-AP gate was not passed.')
    if args.bootstrap:
        groups=[[g.index.to_numpy() for _,g in d.groupby('frame_id')] for _,d in rows.groupby('day')]
        rng=np.random.default_rng(41);y=rows.error.to_numpy();arrays={name:s.to_numpy() for name,s in scores.items()};deltas={name:[] for name in ['baseline','offset_only']}
        for _ in range(args.bootstrap):
            values={name:[] for name in deltas}
            for frames in groups:
                ix=np.concatenate([frames[j] for j in rng.integers(0,len(frames),len(frames))])
                if len(np.unique(y[ix]))<2:break
                candidate=average_precision_score(y[ix],arrays['candidate'][ix])
                for name in deltas:values[name].append(candidate-average_precision_score(y[ix],arrays[name][ix]))
            if all(len(v)==len(groups) for v in values.values()):
                for name in deltas:deltas[name].append(float(np.mean(values[name])))
        result['conditional_frame_bootstrap']={name:dict(delta_percentiles=np.quantile(values,[.025,.5,.975]).tolist(),positive_fraction=float(np.mean(np.asarray(values)>0))) for name,values in deltas.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(pd.DataFrame(metrics).pivot(index=['subset','method'],columns='day',values='AP').to_string())
    if args.bootstrap:print(json.dumps(result['conditional_frame_bootstrap'],indent=2))


if __name__=='__main__':main()
