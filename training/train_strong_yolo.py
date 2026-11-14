"""YOLO26x person teacher with retained COCO classifier and bbox-only adaptation."""
import argparse
import sys
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from sklearn.metrics import average_precision_score

from detector_error_discovery.core import BOX, risks
from diverse_person_data import build
from train_diverse_person import graph

ROOT = Path(__file__).resolve().parents[1]
DAYS = ['20261025', '20261026']
WEIGHTS = None
CONFIG = dict(model='YOLO26x',source='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt',
              epochs=20,training_size=640,inference_size=960,batch=2,seed=619,lr=.0002,
              retained_coco_classes=80,trainable='Detect.cv2 and one2one_cv2 box branches only',
              frozen_classification=True,frozen_batchnorm_statistics=True,cls_loss_weight=0.,
              use_original_GT_and_supported_extra_context=True,pretrained_average=.5)


def digest_state(model,prefixes):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not any(name.startswith(prefix) for prefix in prefixes):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_boxes(trainer):
    index=len(trainer.model.model)-1
    prefixes=[f'model.{index}.cv2.',f'model.{index}.one2one_cv2.']
    for name,param in trainer.model.named_parameters():param.requires_grad_(any(name.startswith(x) for x in prefixes))
    trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    assert trainable>0 and trainer.model.model[-1].nc==80
    trainer._reg_prefixes=prefixes;trainer._fixed_state=digest_state(trainer.model,prefixes)
    print('BBOX_ONLY_TRAINABLE',trainable,'total',sum(p.numel() for p in trainer.model.parameters()),flush=True)


def stop_bn_stats(trainer):
    for module in trainer.model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()


def verify_fixed(trainer):
    actual=digest_state(trainer.model,trainer._reg_prefixes)
    assert actual==trainer._fixed_state,'A frozen classifier/backbone tensor changed during adaptation.'
    print('FROZEN_CLASSIFIER_AND_BACKBONE_UNCHANGED',actual,flush=True)


def infer(weights,rows,path,device):
    if path.exists():return json.loads(path.read_text())
    model=YOLO(str(weights));assert model.names[0]=='person';records=[]
    for number,frame in enumerate(sorted(rows.loc[rows.class_name.eq('person'),'frame_id'].unique()),1):
        pred=model.predict(str(DATA/f'{frame}.jpg'),imgsz=960,conf=.005,iou=.6,classes=[0],device=device,verbose=False)[0]
        records.append(dict(frame_id=frame,boxes=pred.boxes.xyxy.cpu().numpy().tolist(),classes=np.zeros(len(pred.boxes),int).tolist(),confidence=pred.boxes.conf.cpu().numpy().tolist()))
        print('PRED',path.stem,number,'frames',flush=True)
    path.write_text(json.dumps(records));del model;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return records


def train(inventory,day,out,device):
    path=out/f'bbox_{day}.pt'
    fit=inventory if day=='all' else inventory[inventory.day.ne(day)];paths=fit.image.tolist()+fit.loc[fit.source.eq('original')&fit.boxes.gt(0),'image'].tolist()
    assert day=='all' or not fit.day.eq(day).any()
    metadata=dict(CONFIG,holdout=day,training_frames=fit.frame_id.tolist(),training_images=paths,exposures=len(paths),pretrained_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest())
    signature=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest();meta=path.with_suffix('.json')
    if path.exists():assert json.loads(meta.read_text())['signature']==signature;return path
    listing=out/f'fit_{day}.txt';listing.write_text('\n'.join(paths));model=YOLO(str(WEIGHTS));names=model.names
    config=out/f'fit_{day}.yaml';config.write_text(yaml.safe_dump(dict(path=str(out/'dataset'),train=str(listing),val=str(listing),names=names)))
    model.add_callback('on_train_start',freeze_boxes);model.add_callback('on_train_batch_start',stop_bn_stats);model.add_callback('on_train_end',verify_fixed)
    run=f'run_bbox_{day}';print('FIT STRONG',day,'exposures',len(paths),flush=True)
    model.train(data=str(config),epochs=20,imgsz=640,batch=2,workers=0,device=device,freeze=0,optimizer='AdamW',lr0=.0002,lrf=.1,weight_decay=.0001,mosaic=0.,degrees=3,scale=.1,fliplr=.5,seed=619,deterministic=True,patience=20,val=False,plots=False,amp=False,cls=0.,box=7.5,project=str(out),name=run,exist_ok=True)
    shutil.copy2(out/run/'weights/last.pt',path);meta.write_text(json.dumps(dict(signature=signature,**metadata),indent=2));del model;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return path


def average(weights,out):
    path=weights.with_stem(weights.stem+'_prior50')
    if path.exists():return path
    prior=YOLO(str(WEIGHTS)).model.float();adapted=YOLO(str(weights)).model.float();state=adapted.state_dict();initial=prior.state_dict()
    for name in state:
        if '.cv2.' in name or '.one2one_cv2.' in name:
            if state[name].is_floating_point():state[name]=.5*state[name]+.5*initial[name]
    adapted.load_state_dict(state);checkpoint=torch.load(weights,map_location='cpu',weights_only=False);checkpoint['model']=adapted.half();checkpoint['ema']=None;checkpoint['optimizer']=None;torch.save(checkpoint,path);return path


def evaluate(rows,variants,out):
    person=rows.class_name.eq('person');p=rows.p_clothes.fillna(0);baseline=graph(rows);scores={'reference_public_087853':baseline}
    for name,records in variants.items():
        for matching in ['legacy','exact']:
            risk=risks(rows.loc[person],records,names=['person'],threshold=.1,legacy=matching=='legacy');scores[name+'_'+matching+'_replacement']=graph(rows,risk)
            # A smaller prior-preserving correction, separate from graph replacement.
            s=baseline.copy();protected=risk+p.loc[person]*(1-risk);s.loc[person]=.75*s.loc[person]+.25*protected;scores[name+'_'+matching+'_blend25']=s
        independent=risks(rows.loc[person],records,names=['person'],threshold=.3,legacy=False)
        for weight in [1.,.5]:
            inputs=rows.copy();inputs.loc[person,'grounded_person']=(1-weight)*rows.loc[person,'grounded_person']+weight*independent
            scores[name+f'_independent_w{weight}']=graph(inputs)
    report=[]
    for day,g in rows.groupby('day'):
        for subset in ['all','person','test_like']:
            ix=g.index if subset=='all' else g.index[g.class_name.eq('person')] if subset=='person' else g.index[g.class_name.ne('person')|g.confidence.ge(.73)]
            for name,s in scores.items():report.append(dict(day=day,subset=subset,method=name,AP=average_precision_score(rows.loc[ix,'error'],s.loc[ix])))
    report=pd.DataFrame(report);report.to_csv(out/'validation.csv',index=False);pd.DataFrame(scores).assign(id=rows.id).to_csv(out/'oof_scores.csv',index=False);tables={}
    for subset in report.subset.unique():
        t=report[report.subset.eq(subset)].pivot(index='method',columns='day',values='AP');t['mean']=t.mean(axis=1);t.to_csv(out/f'{subset}_summary.csv');tables[subset]=t;print(subset+'\n'+t.sort_values('mean',ascending=False).to_string(),flush=True)
    a=tables['all'];pt=tables['person'];ref='reference_public_087853';eligible=(a[DAYS]>=a.loc[ref,DAYS]).all(axis=1)&a['mean'].ge(a.loc[ref,'mean']+.002)&(pt[DAYS]>=pt.loc[ref,DAYS]).all(axis=1)&pt['mean'].ge(pt.loc[ref,'mean']+.01)&(tables['test_like'][DAYS]>=tables['test_like'].loc[ref,DAYS]).all(axis=1)
    chosen=a.loc[eligible,'mean'].idxmax() if eligible.any() else None
    result=dict(chosen=chosen,public_AP=None,config=CONFIG,default_changed=False,criterion='Both days all/person/test-like nondecreasing; mean all +.002, person +.01.',caveat='Exploratory on two reused days; not an independent estimate of Public AP.')
    (out/'selection.json').write_text(json.dumps(result,indent=2));print('SELECTED',chosen,flush=True);return chosen


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['frozen','cv','evaluate'],default='cv');parser.add_argument('--weights',type=Path);parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'outputs/strong_yolo_person');args=parser.parse_args();
    from detector_error_discovery.strong_person import download
    WEIGHTS=args.weights if args.weights is not None else download(ROOT)
    DATA=args.data_dir;out=args.output;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);device='mps' if torch.backends.mps.is_available() else 'cpu';rows=pd.read_csv(ROOT/'reference/diverse_person_graph_inputs.csv',dtype={'id':str,'day':str});variants={}
    variants['frozen26x']=infer(WEIGHTS,rows,out/'frozen26x_train.json',device)
    evaluate(rows,variants,out)
    if args.stage=='frozen':sys.exit(0)
    inventory,_=build(DATA,out)
    for day in DAYS:
        if args.stage!='evaluate':path=train(inventory,day,out,device)
        else:path=out/f'bbox_{day}.pt'
        for name,weights in [('bbox26x',path),('prior50',average(path,out))]:
            records=infer(weights,rows[rows.day.eq(day)],out/f'{name}_{day}_oof.json',device);variants.setdefault(name,[]).extend(records)
    evaluate(rows,variants,out)
