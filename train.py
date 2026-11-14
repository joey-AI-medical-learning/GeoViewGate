# coding=utf-8
import argparse
import os
import time
import logging
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

from dataset.heterogeneous import Teeth_load_all_train_mil, Teeth_load_all_test_mil
from dataset.heterogeneous_sampling import SamplingConfig

from dataset.data_utils import init_fn
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader


from modal import Stage_2
from utils.evaluation_utils import ATTENTION, CASE_FIELDS, METRICS, case_rows, evaluate_batches, summarize_cases, write_csv



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(PROJECT_DIR, 'excel')
DEFAULT_SAMPLING = 'heterogeneous_v1'
DEFAULT_DATA_ROOT = '/gemini/data-1/data'
DEFAULT_PRETRAIN = os.path.normpath(os.path.join(
    PROJECT_DIR, '..', 'Pretrain', 'output', 'pretrain', 'pretrained_encoder.pth'))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, 'output')

def configure(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--datapath', default=DEFAULT_DATA_ROOT, type=str)
    parser.add_argument('--savepath', default=None, type=str,
                    help='By default, results are saved to the project output/seed_<seed> directory.')
    parser.add_argument('--pretrain', default=DEFAULT_PRETRAIN, type=str)
    parser.add_argument('--model', default='GeoViewGate', type=str) 
    parser.add_argument('--num_instances', default=4, type=int, help='Number of patches (K) sampled per patient for MIL.')
    parser.add_argument('--sampling', choices=['heterogeneous_v1'], default=DEFAULT_SAMPLING,
                    help='128-pixel main field of view + thin slices at the upper and lower boundaries + 32-pixel texture field of view.')
    parser.add_argument('--slab-axis', type=int, choices=[0,1,2], default=2)
    parser.add_argument('--slab-thickness', type=int, choices=range(15,27), default=26)
    parser.add_argument('--slab-overlap', type=int, default=13, help='Number of slices extending inward from the MASK bounding box along the depth axis; default: 13 slices inward and 13 slices outward.')
    parser.add_argument('--extra-width', type=int, choices=range(7, 21), default=12, help='Number of voxels to expand on each side; the total increase in length and width is twice this value.')
    parser.add_argument('--lr' , default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--validate-start-epoch', type=int, default=50, help='Preserve the original setting: 50 means validation starts at epoch 51; set to 0 for quick checks.')
    parser.add_argument('--seed', default=1024, type=int)  

    args = parser.parse_args(argv)
    if args.pretrain and args.pretrain.lower() in ('none', 'null'):
        args.pretrain = None
    if args.num_instances != 4:
        parser.error('This experiment requires exactly four heterogeneous views')
    sampling_config = SamplingConfig(slab_axis=args.slab_axis, thickness=args.slab_thickness, overlap=args.slab_overlap,
                                     extra_width=args.extra_width)
    if args.savepath is None:
        args.savepath = os.path.join(OUTPUT_ROOT, f'seed_{args.seed}')
    if not os.path.isdir(args.datapath):
        parser.error(f'Data directory does not exist: {args.datapath}. Please update DEFAULT_DATA_ROOT in the main script.')
    if args.pretrain is not None and not os.path.isfile(args.pretrain):
        parser.error(f'Pretrained checkpoint not found: {args.pretrain}. Please update DEFAULT_PRETRAIN in the main script.')
    if args.batch_size < 1 or args.num_epochs < 1 or args.num_workers < 0:
        parser.error('batch_size/num_epochs must be positive; num-workers must be nonnegative')
    if args.validate_start_epoch < 0:
        parser.error('validate-start-epoch must be nonnegative')
    for split_name in ('7.xlsx', '1.xlsx', '2.xlsx'):
        if not os.path.isfile(os.path.join(SPLIT_DIR, split_name)):
            parser.error(f'Missing project Excel: {os.path.join(SPLIT_DIR, split_name)}')
    if os.path.isdir(args.savepath) and os.listdir(args.savepath):
        parser.error(f'Training results already exist. Please specify a different --savepath: {args.savepath}')
    setup(args, 'training')
    if args.validate_start_epoch >= args.num_epochs:
        logging.warning('No validation is performed during this run; no model_earlystop checkpoint will be selected. Set --validate-start-epoch 0 to enable validation for short runs.')
    logging.info(f'Configuration: sampling={args.sampling}, data={args.datapath}, ')
                 f'pretrain={args.pretrain}, output={args.savepath}')

    args.train_transforms = 'Compose([RandomIntensityChange((0.1,0.1)), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

    return args, sampling_config

REPORT_FIELDS = ('num_cases', *METRICS, 'loss_ce', 'attn_max', 'attn_entropy',
                 *ATTENTION,
                 *(f'class_{label}_{metric}' for label in range(3)
                   for metric in ('support', 'precision', 'recall', 'f1')))
EPOCH_FIELDS = ('epoch', 'lr', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                'train_class_loss', 'grad_norm', 'epoch_time_min',
                'gpu_mem_gb', 'best_val_epoch', 'early_stopped',
                *(f'{split}_{key}' for split in ('train', 'val') for key in REPORT_FIELDS))


def report_columns(split, report):
    columns = {f'{split}_{key}': report.get(key) for key in REPORT_FIELDS}
    for key, value in report['mean_attention'].items():
        columns[f'{split}_{key}'] = value
    for label, metrics in report['per_class'].items():
        for key, value in metrics.items():
            columns[f'{split}_class_{label}_{key}'] = value
    return columns


def main(argv=None):
    args, sampling_config = configure(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpts = args.savepath
    with open(os.path.join(ckpts, 'run_config.json'), 'w', encoding='utf-8') as stream:
        json.dump({'arguments':vars(args), 'sampling_config':sampling_config.to_dict(),
                   'loss':'cross_entropy(label_smoothing=0.1)'},
                  stream, ensure_ascii=False, indent=2)
    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

    csv_path = os.path.join(ckpts, 'train_log.csv')
    train_cases_path = os.path.join(ckpts, 'train_cases.csv')
    val_cases_path = os.path.join(ckpts, 'val_cases.csv')
    case_fields = ('epoch', *CASE_FIELDS)
    write_csv(csv_path, EPOCH_FIELDS)
    write_csv(train_cases_path, case_fields)
    write_csv(val_cases_path, case_fields)
    logging.info('CSV files: %s, %s, %s', csv_path, train_cases_path, val_cases_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    model = Stage_2.Model(in_chans=3, num_classes=3)


    if args.pretrain is not None:
        sd = torch.load(args.pretrain, map_location='cpu', weights_only=True)
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        backbone_missing = [key for key in missing if not key.startswith('head.')]
        if backbone_missing or unexpected:
            raise ValueError(f'Pretrained backbone mismatch: missing={backbone_missing}, unexpected={unexpected}')
        logging.info(f'Loaded pretrained encoder: {args.pretrain} (missing={len(missing)}, unexpected={len(unexpected)})')

    model = torch.nn.DataParallel(model.to(device), device_ids=[0])

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    class_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_file = os.path.join(SPLIT_DIR, '7.xlsx')
    val_file = os.path.join(SPLIT_DIR, '1.xlsx')

    logging.info(str(args))
    logging.info('Heterogeneous sampling: %s', sampling_config.to_dict())
    train_set = Teeth_load_all_train_mil(transforms=args.train_transforms, root=args.datapath,
                                       train_file=train_file, num_instances=args.num_instances, config=sampling_config)
    val_set = Teeth_load_all_test_mil(transforms=args.test_transforms, root=args.datapath,
                                     test_file=val_file, num_instances=args.num_instances, config=sampling_config)
    if set(train_set.names) & set(val_set.names):
        raise ValueError('Training and validation patient IDs overlap')
    test_file = os.path.join(SPLIT_DIR, '2.xlsx')
    if os.path.isfile(test_file):
        import pandas as pd
        test_ids = set(pd.read_excel(test_file, dtype={'patient_id':str}).patient_id)
        if test_ids & (set(train_set.names) | set(val_set.names)):
            raise ValueError('Held-out test patient IDs overlap with training or validation')

    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=max(1, args.batch_size // args.num_instances),
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        worker_init_fn=init_fn)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + 1),
        pin_memory=True)

    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')

    best_val_loss = float('inf')
    best_val_epoch = None
    no_improve_count = 0
    early_stop_patience = 20
    validate_start_epoch = args.validate_start_epoch

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        lr_schedule(optimizer, epoch)
        epoch_loss = 0.0
        train_rows = []
        epoch_grad_norm = 0.0
        grad_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", ncols=120)

        for x, class_label, name in pbar:
            x = model_mil.move_to_device(x, device)
            class_label = class_label.to(device)

            class_output, attention = model(x)
            class_cross_loss = class_loss(class_output, class_label)
            loss = class_cross_loss

            optimizer.zero_grad()
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            epoch_grad_norm += total_norm.item()
            grad_steps += 1
            optimizer.step()

            epoch_loss += loss.item() * class_label.size(0)
            train_rows.extend(case_rows(class_output, class_label, attention, name))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        number = epoch + 1
        train_report = summarize_cases(train_rows)
        write_csv(train_cases_path, case_fields,
                  ({'epoch': number, **row} for row in train_rows), append=True)
        avg_train_loss = epoch_loss / len(train_rows)
        avg_grad_norm = epoch_grad_norm / max(1, grad_steps)
        metrics = {
            'epoch': number, 'lr': optimizer.param_groups[0]['lr'],
            'train_loss': avg_train_loss, 'train_acc': train_report['accuracy'],
            'train_class_loss': train_report['loss_smoothed'],
            'grad_norm': avg_grad_norm,
            **report_columns('train', train_report),
        }
        if writer is not None:
            writer.add_scalar('loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('grad_norm', avg_grad_norm, epoch)
        logging.info('Epoch %d/%d, Train Loss=%.4f, Train Acc=%.4f',
                     number, args.num_epochs, avg_train_loss, train_report['accuracy'])

        checkpoint = {
            'epoch': number, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict(),
            'sampling': args.sampling, 'sampling_config': sampling_config.to_dict(), 'architecture': 'heterogeneous_mil',
        }
        if number % 20 == 0 or number > args.num_epochs - 5:
            torch.save(checkpoint, os.path.join(ckpts, f'model_epoch_{number}.pth'))
        torch.save(checkpoint, os.path.join(ckpts, 'model_last.pth'))

        if number > validate_start_epoch:
            val_rows = [row for batch in evaluate_batches(model, val_loader) for row in batch]
            val_report = summarize_cases(val_rows)
            val_loss, val_acc = val_report['loss_smoothed'], val_report['accuracy']
            write_csv(val_cases_path, case_fields,
                      ({'epoch': number, **row} for row in val_rows), append=True)
            metrics.update(val_loss=val_loss, val_acc=val_acc, **report_columns('val', val_report))
            if writer is not None:
                writer.add_scalar('loss/val_epoch', val_loss, epoch)
                writer.add_scalar('acc/val_epoch', val_acc, epoch)
            logging.info('Epoch %d, Val Loss=%.4f, Val Acc=%.4f', number, val_loss, val_acc)
            if val_loss < best_val_loss:
                best_val_loss, best_val_epoch = val_loss, number
                no_improve_count = 0
                torch.save(checkpoint, os.path.join(ckpts, 'model_earlystop.pth'))
                logging.info('Saved best validation checkpoint from epoch %d', number)
            else:
                no_improve_count += 1
                logging.info('No improvement for %d/%d epochs', no_improve_count, early_stop_patience)

        stopped = no_improve_count >= early_stop_patience
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        metrics.update(best_val_epoch=best_val_epoch, early_stopped=int(stopped),
                       epoch_time_min=(time.time() - epoch_start) / 60, gpu_mem_gb=gpu_mem_gb)
        write_csv(csv_path, EPOCH_FIELDS, [metrics], append=True)
        if stopped:
            logging.info('Early stopping at epoch %d', number)
            break

    if writer is not None:
        writer.close()
    total_hours = (time.time() - start) / 3600
    logging.info(f"Total Training Time: {total_hours:.4f} hours")


if __name__ == '__main__':
    main()
