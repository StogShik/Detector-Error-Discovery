"""Frozen SAM2.1-tiny masks: visible-body spillover and neighboring instances.

Prompts use proposal boxes and frozen Grounding DINO detections only. Original
GT is used only as held-out labels/visualization, never as a segmentation prompt.
Cross-day classifiers use frozen signals only (no OOF ranker inputs in fitting).
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detector_error_discovery.core import BOX, overlaps, assignment

REPO=Path(__file__).resolve().parents[1]
REFERENCE=REPO/'reference'
OUT=REPO/'outputs/sam2_person'
ROOT=None
DAYS=['20261025','20261026']

def load_rows():
    return pd.read_csv(REFERENCE/'person_recipe_oof.csv',dtype={'id':str,'day':str})

CONFIG=dict(model='SAM 2.1 Hiera Tiny',checkpoint_url='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
    source='https://github.com/facebookresearch/sam2',config='configs/sam2.1/sam2.1_hiera_t.yaml',
    frozen=True,version=2,multimask=True,selection='Highest SAM predicted mask quality; no GT selection',
    prompts=['original proposal box','independent frozen GDI box','two foreground torso points and neighboring negative points'],
    fitting='Other UTC day only; frozen public signals, no supervised ranker features in fitting',
    logistic_C=.1,minimum_teacher_confidence=.3)
BASE_COLS=['confidence','cx','cy','logw','logh','teacher_iou','teacher_confidence','teacher_found']
MASK_COLS=['proposal_box_iou','proposal_outside','proposal_fill','proposal_quality',
    'teacher_box_iou','teacher_outside','teacher_fill','teacher_quality',
    'points_box_iou','points_outside','points_quality','mask_agreement','neighbor_occupancy','neighbor_in_mask']


def mask_box(mask):
    y,x=np.where(mask)
    return np.array([x.min(),y.min(),x.max()+1,y.max()+1],float) if len(x) else np.zeros(4)


def inside(mask,box):
    x1,y1,x2,y2=np.clip(np.rint(box),0,[1280,720,1280,720]).astype(int)
    return float(mask[y1:y2,x1:x2].sum())


def mask_iou(a,b):
    return float(np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1))


def describe(mask,quality,box):
    area=float(mask.sum());intersection=inside(mask,box);wh=np.maximum(box[2:]-box[:2],1);extent=mask_box(mask)
    return dict(box_iou=float(overlaps(box[None],extent[None])[0,0]),outside=1-intersection/max(area,1),
        fill=intersection/wh.prod(),quality=float(quality),bottom=(extent[3]-box[3])/wh[1],
        top=(box[1]-extent[1])/wh[1],mask_area_ratio=area/wh.prod())


def select(predictor,**kwargs):
    masks,quality,_=predictor.predict(multimask_output=True,**kwargs);best=int(np.argmax(quality))
    return masks[best].astype(bool),float(quality[best])


def infer(split):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    rows=load_rows();people=rows.loc[rows.class_name.eq('person')]
    raw={d['frame_id']:d for d in json.loads((REFERENCE/'sam2_person_gdi_train.json').read_text())}
    path=OUT/f'{split}_features.csv';cached=pd.read_csv(path,dtype={'id':str,'day':str}) if path.exists() else pd.DataFrame();done=set(cached.id) if len(cached) else set()
    signature=hashlib.sha256((people[['id','frame_id',*BOX]].to_csv(index=False)+json.dumps(CONFIG,sort_keys=True)).encode()).hexdigest()
    metadata=OUT/f'{split}_metadata.json';checkpoint=OUT/'sam2.1_hiera_tiny.pt'
    if not checkpoint.exists():
        from urllib.request import urlretrieve
        urlretrieve(CONFIG['checkpoint_url'],checkpoint)
    expected=json.loads((REFERENCE/'sam2_person_metadata.json').read_text())['checkpoint_sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    meta=dict(CONFIG,input_sha256=signature,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if metadata.exists():assert json.loads(metadata.read_text())==meta
    else:metadata.write_text(json.dumps(meta,indent=2))
    if done==set(people.id):return cached
    torch.set_num_threads(3);device='mps' if torch.backends.mps.is_available() else 'cpu'
    print('SAM LOAD',device,flush=True);predictor=SAM2ImagePredictor(build_sam2(CONFIG['config'],str(checkpoint),device=device));records=cached.to_dict('records')
    for number,(frame,group) in enumerate(people.groupby('frame_id'),1):
        if set(group.id)<=done:continue
        d=raw[frame];boxes=np.asarray(d['boxes']).reshape(-1,4);conf=np.asarray(d['confidence']);keep=(np.asarray(d['classes'])==0)&(conf>=.3);boxes=boxes[keep];conf=conf[keep]
        overlap_matrix=overlaps(group[BOX].to_numpy(float),boxes);pairs=assignment(overlap_matrix,soft=True)
        with Image.open(ROOT/f'{frame}.jpg') as im:image=np.asarray(im.convert('RGB')).copy()
        assert image.shape[:2]==(720,1280)
        with torch.inference_mode():
            predictor.set_image(image)
            teachers=[select(predictor,box=box) for box in boxes]
            for i,(_,row) in enumerate(group.iterrows()):
                if row.id in done:continue
                box=row[BOX].to_numpy(float);wh=np.maximum(box[2:]-box[:2],1);j=pairs.get(i);found=j is not None and overlap_matrix[i,j]>=.1
                target=boxes[j] if found else box
                positive=np.array([[(target[0]+target[2])/2,target[1]+.25*(target[3]-target[1])],[(target[0]+target[2])/2,target[1]+.5*(target[3]-target[1])]])
                negatives=np.array([[(b[0]+b[2])/2,b[1]+.25*(b[3]-b[1])] for k,b in enumerate(boxes) if (not found or k!=j)]).reshape(-1,2)
                # Avoid contradictory positive/negative prompts at overlaps.
                if len(negatives):negatives=negatives[np.linalg.norm(negatives[:,None]-positive[None],axis=2).min(axis=1)>20]
                points=np.r_[positive,negatives];labels=np.r_[np.ones(len(positive),int),np.zeros(len(negatives),int)]
                proposal_mask,proposal_quality=select(predictor,box=box)
                teacher_mask,teacher_quality=teachers[j] if found else (proposal_mask,proposal_quality)
                points_mask,points_quality=select(predictor,point_coords=points,point_labels=labels)
                record=dict(id=row.id,frame_id=frame,day=row.day,confidence=row.confidence,
                    cx=(box[0]+box[2])/2560,cy=(box[1]+box[3])/1440,logw=np.log(wh[0]/1280),logh=np.log(wh[1]/720),
                    teacher_iou=float(overlap_matrix[i,j]) if found else 0,teacher_confidence=float(conf[j]) if found else 0,teacher_found=int(found))
                for mode,mask,q in [('proposal',proposal_mask,proposal_quality),('teacher',teacher_mask,teacher_quality),('points',points_mask,points_quality)]:record.update({mode+'_'+k:v for k,v in describe(mask,q,box).items()})
                other=[m for k,(m,q) in enumerate(teachers) if (not found or k!=j) and q>=.7]
                neighbor=np.logical_or.reduce(other) if len(other) else np.zeros((720,1280),bool)
                record['mask_agreement']=mask_iou(proposal_mask,teacher_mask);record['neighbor_occupancy']=inside(neighbor&~teacher_mask,box)/wh.prod()
                inside_seed=[bool(teacher_mask[int(np.clip(y,0,719)),int(np.clip(x,0,1279))]) for x,y in negatives];record['neighbor_in_mask']=float(np.mean(inside_seed)) if len(inside_seed) else 0.
                records.append(record);done.add(row.id)
                np.savez_compressed(OUT/f'{split}_{row.id}_masks.npz',masks=np.packbits(np.stack([proposal_mask,teacher_mask,points_mask]),axis=-1),qualities=[proposal_quality,teacher_quality,points_quality],shape=[720,1280])
        pd.DataFrame(records).to_csv(path,index=False);print(f'SAM {split} {number}/{people.frame_id.nunique()} frames, {len(done)}/{len(people)} proposals',flush=True)
    del predictor;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return pd.DataFrame(records)


def learn(train,features,holdout):
    fit=train.loc[train.day.ne(holdout)] if holdout!='all' else train
    assert holdout=='all' or not fit.day.eq(holdout).any()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=192))
    model.fit(fit[features],fit.error);return model


def candidates(rows,baseline,features,models=None):
    person=rows.class_name.eq('person');part=rows.loc[person];f=features.set_index('id').loc[part.id].reset_index();f.index=part.index
    scores={}
    for mode in ['proposal','teacher','points']:
        risk=expit((.6-f[mode+'_box_iou'])/.12);quality=f[mode+'_quality'].clip(0,1)
        s=baseline.copy();s.loc[person]=.75*baseline.loc[person]+.25*(quality*risk+(1-quality)*baseline.loc[person]);scores[mode+'_bbox_w0.25']=s
    outside=(np.clip((f.teacher_outside-.05)/.25,0,1)*f.teacher_quality.clip(0,1)).to_numpy();neighbor=np.clip(f.neighbor_occupancy.to_numpy()/.3,0,1);evidence=np.maximum(outside,neighbor)
    for w in [.25,.5]:
        s=baseline.copy();s.loc[person]+=w*evidence*(1-s.loc[person]);scores[f'spillover_neighbor_w{w}']=s
    for kind,columns in [('control',BASE_COLS),('mask',BASE_COLS+MASK_COLS)]:
        risk=pd.Series(index=part.index,dtype=float)
        for day,g in part.groupby('day'):
            model=models[(kind,day)] if models is not None else learn(features,columns,day)
            risk.loc[g.index]=model.predict_proba(f.loc[g.index,columns])[:,1]
        for w in [.25,.5]:
            s=baseline.copy();s.loc[person]=(1-w)*s.loc[person]+w*risk;scores[f'{kind}_logistic_w{w}']=s
    return scores


def evaluate(features):
    rows=load_rows();baseline=rows.baseline_public_model
    features=features.copy();features['error']=features.id.map(rows.set_index('id').error)
    models={(kind,day):learn(features,cols,day) for kind,cols in [('control',BASE_COLS),('mask',BASE_COLS+MASK_COLS)] for day in DAYS}
    scores={'reference_public_087853':baseline,**candidates(rows,baseline,features,models)};report=[]
    for day,g in rows.groupby('day'):
        for subset in ['all','person','test_like','person_test_like']:
            idx=g.index if subset=='all' else g.index[g.class_name.eq('person')] if subset=='person' else g.index[g.class_name.ne('person')|g.confidence.ge(.73)] if subset=='test_like' else g.index[g.class_name.eq('person')&g.confidence.ge(.73)]
            for name,s in scores.items():report.append(dict(day=day,subset=subset,method=name,AP=average_precision_score(rows.loc[idx,'error'],s.loc[idx])))
    report=pd.DataFrame(report);report.to_csv(OUT/'validation.csv',index=False);pd.DataFrame(scores).assign(id=rows.id).to_csv(OUT/'oof_scores.csv',index=False);tables={}
    for subset in report.subset.unique():
        table=report.loc[report.subset.eq(subset)].pivot(index='method',columns='day',values='AP');table['mean']=table.mean(axis=1);table.to_csv(OUT/f'{subset}_summary.csv');tables[subset]=table;print(subset+'\n'+table.sort_values('mean',ascending=False).to_string(),flush=True)
    a=tables['all'];p=tables['person'];ref='reference_public_087853';valid=(a[DAYS]>=a.loc[ref,DAYS]).all(axis=1)&a['mean'].ge(a.loc[ref,'mean']+.002)&(p[DAYS]>=p.loc[ref,DAYS]).all(axis=1)&p['mean'].ge(p.loc[ref,'mean']+.01)&(tables['test_like'][DAYS]>=tables['test_like'].loc[ref,DAYS]).all(axis=1)
    for name in valid.index:
        if name.startswith('control'):valid.loc[name]=False
        if name.startswith('mask_logistic'):
            ctrl=name.replace('mask_','control_');valid.loc[name]&=a.loc[name,'mean']>a.loc[ctrl,'mean']+.001
    chosen=a.loc[valid,'mean'].idxmax() if valid.any() else None
    selection=dict(chosen=chosen,public_AP=None,reference_public_AP=.87853,default_changed=False,
        criterion='Both days all/person/test-like AP nondecreasing; mean all gain >=.002 and person >=.01; learned mask must exceed frozen-detector-only control mean all AP by .001.',
        caveat='Exploratory on two repeatedly reused days, not an independent Public estimate.')
    if chosen:
        if chosen.startswith('mask_logistic'):
            for kind,columns in [('control',BASE_COLS),('mask',BASE_COLS+MASK_COLS)]:joblib.dump(learn(features,columns,'all'),OUT/f'{kind}_all.joblib')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTED',chosen,flush=True)



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer',action='store_true',help='Regenerate masks from original TRAIN images')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--sam-source',type=Path,help='Optional checkout of the pinned official SAM2 source')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    if args.infer:
        if args.data_dir is None:parser.error('--infer requires --data-dir')
        ROOT=args.data_dir
        if args.sam_source is not None:sys.path.insert(0,str(args.sam_source))
        values=infer('train')
    else:values=pd.read_csv(REFERENCE/'sam2_person_features.csv',dtype={'id':str,'day':str})
    assert len(values)==84 and values.id.is_unique
    evaluate(values)
