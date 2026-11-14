"""Frozen public Objects365->COCO D-FINE-X as an independent person verifier."""
import argparse,json,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from transformers import AutoImageProcessor,DFineForObjectDetection
from train_strong_yolo import evaluate

ROOT=Path(__file__).resolve().parents[1]
MODEL='ustc-community/dfine-xlarge-obj2coco'
REVISION='15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d'


def infer(rows,data,out,split='train'):
 path=out/f'dfine_{split}.json';records=json.loads(path.read_text()) if path.exists() else [];seen={r['frame_id'] for r in records};frames=sorted(rows.loc[rows.class_name.eq('person'),'frame_id'].unique())
 if seen==set(frames):return records
 torch.set_num_threads(2)
 processor=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION,use_fast=False);model=DFineForObjectDetection.from_pretrained(MODEL,revision=REVISION).eval().requires_grad_(False).to('cpu');person=[int(k) for k,v in model.config.id2label.items() if v=='person'];assert len(person)==1
 for number,frame in enumerate(frames,1):
  if frame in seen:continue
  with Image.open(data/f'{frame}.jpg') as source:image=source.convert('RGB')
  inputs=processor(images=image,return_tensors='pt')
  with torch.inference_mode():outputs=model(**inputs)
  pred=processor.post_process_object_detection(outputs,target_sizes=torch.tensor([[image.height,image.width]]),threshold=.01)[0];select=pred['labels'].eq(person[0]);b=pred['boxes'][select].cpu().numpy();p=pred['scores'][select].cpu().numpy();records.append(dict(frame_id=frame,boxes=b.tolist(),confidence=p.tolist(),classes=np.zeros(len(b),int).tolist()));path.write_text(json.dumps(records));print('DFINE_CPU',number,'/',len(frames),flush=True)
 del model;gc.collect();return records

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'outputs/dfine_person');a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=pd.read_csv(ROOT/'reference/diverse_person_graph_inputs.csv',dtype={'id':str,'day':str});raw=infer(rows,a.data_dir,a.output);evaluate(rows,{'dfine_x':raw},a.output)
 p=a.output/'selection.json';m=json.loads(p.read_text());m['config']=dict(model=MODEL,revision=REVISION,model_fitting='Frozen public weights; no local fitting',source='https://huggingface.co/'+MODEL);p.write_text(json.dumps(m,indent=2))
