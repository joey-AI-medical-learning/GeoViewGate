import argparse
import logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset.heterogeneous import Brats_load_all_test_mil
from dataset.heterogeneous_sampling import SamplingConfig
from utils.evaluation_utils import evaluate
from model import Stage_2

PROJECT=Path(__file__).resolve().parent
SPLIT_DIR=PROJECT/'excel'
EXPECTED_SAMPLING='heterogeneous_v1'


def validate_test_mil(model,test_loader,csv_name=None,metrics_path=None,**kwargs):
    return evaluate(model,test_loader,csv_name,metrics_path,**kwargs)


def main(argv=None,forced_split=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',nargs='?',default='model_earlystop.pth')
    parser.add_argument('--checkpoint-dir',type=Path,default=None)
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--datapath',type=Path,default=Path('/gemini/data-1/data'))
    parser.add_argument('--sampling',choices=[EXPECTED_SAMPLING],default=EXPECTED_SAMPLING)
    parser.add_argument('--split',choices=['val','test'],default='test')
    args=parser.parse_args(argv)
    if forced_split:
        args.split=forced_split
    if args.checkpoint_dir is None:
        args.checkpoint_dir=PROJECT/'output'/'seed_1024'
    checkpoint_path=(args.checkpoint_dir/args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint not found: {checkpoint_path}')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint.get('sampling') != EXPECTED_SAMPLING:
        parser.error('Model version mismatch. Please use the evaluation code from the project associated with this checkpoint.')
    if checkpoint.get('use_fcem', False):
        parser.error('This checkpoint contains FCEM. This standalone project only supports the original without_fcem checkpoints or checkpoints trained within this project.')
    config=SamplingConfig.from_dict(checkpoint['sampling_config'])
    split_file=SPLIT_DIR / ('1.xlsx' if args.split=='val' else '2.xlsx')
    out = args.output_dir or args.checkpoint_dir
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.checkpoint).stem
    if base.startswith('model_'):
        base = base[len('model_'):]
    prefix = 'val_' if args.split == 'val' else ''
    transforms='Compose([NumpyType((np.float32,np.int64)),])'
    dataset=Brats_load_all_test_mil(root=str(args.datapath),test_file=split_file,transforms=transforms,config=config)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=torch.nn.DataParallel(Stage_2.Model().to(device),device_ids=[0])
    model.load_state_dict(checkpoint['state_dict'])
    train_file=SPLIT_DIR/'7.xlsx'
    train_ids=set(pd.read_excel(train_file,dtype={'patient_id':str}).patient_id) if train_file.is_file() else None
    overlap=sorted(set(dataset.names)&train_ids) if train_ids is not None else None
    metadata={'split':args.split,'split_file':str(split_file.resolve()),
              'checkpoint':str(checkpoint_path),'checkpoint_epoch':checkpoint.get('epoch'),
              'sampling_config':config.to_dict(),'architecture':'heterogeneous_mil',
              'train_id_overlap':overlap,
              'class_counts':{str(i):dataset.labels.count(i) for i in range(3)}}
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(out / f'{prefix}evaluate_{base}.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    start = f'Split={args.split}; file={split_file}; saved epoch={checkpoint.get("epoch")}; sampling={config.to_dict()}'
    print(start, flush=True)
    logger.info(start)
    if overlap:
        print(f'Warning: {len(overlap)} patient IDs overlap with the training set. See the metrics JSON for details.', flush=True)
        logger.warning('Overlapping training IDs: %s', overlap)
    try:
        return evaluate(model, loader, out / f'{prefix}result_{base}.csv',
                        out / f'{prefix}metrics_{base}.json', metadata=metadata)
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__=='__main__':
    main()
