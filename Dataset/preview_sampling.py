import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT))
import numpy as np
from dataset.heterogeneous_sampling import SamplingConfig, sample_views, crop_padded

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--volume',required=True)
    parser.add_argument('--mask',required=True)
    parser.add_argument('--output',default=str(PROJECT/'output'/'sampling_preview'))
    parser.add_argument('--slab-axis',type=int,choices=[0,1,2],default=2)
    parser.add_argument('--slab-thickness',type=int,choices=range(15,27),default=26)
    parser.add_argument('--slab-overlap',type=int,default=13)
    parser.add_argument('--extra-width',type=int,choices=range(7,21),default=12)
    args=parser.parse_args()
    image=np.load(args.volume,allow_pickle=False)
    mask=np.load(args.mask,allow_pickle=False)
    cfg=SamplingConfig(slab_axis=args.slab_axis,thickness=args.slab_thickness,overlap=args.slab_overlap,extra_width=args.extra_width)
    views,meta=sample_views(image,mask,cfg)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    for (x,y),entry in zip(views,meta['views']):
        native_x,native_y=crop_padded(image,mask,entry['start'],entry['native_shape'])
        for suffix,a,b in [('native',native_x,native_y),('input',x,y)]:
            np.save(output/(entry['name']+'_'+suffix+'.npy'),np.concatenate([a,b[...,None]],axis=-1))
    (output/'geometry.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(output.resolve())

if __name__=='__main__':
    main()
