import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datapath', default='/gemini/data-1/data')
    parser.add_argument('--pretrain', default=str(PROJECT.parent/'Pretrain'/'output'/'pretrain'/'pretrained_encoder.pth'))
    parser.add_argument('--output-root', type=Path, default=PROJECT/'output')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4, help='Original patch budget: 4 means one case with four views.')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--slab-axis', type=int, choices=[0,1,2], default=2)
    parser.add_argument('--validate-start-epoch', type=int, default=50)
    parser.add_argument('--stage', choices=['all','train','test'], default='all')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32-1:
        parser.error('seed must be within 0..2**32-1')
    if args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error('Invalid batch size, workers, learning rate, or weight decay')
    if args.stage != 'test' and not 0 <= args.validate_start_epoch < args.num_epochs:
        parser.error('validate-start-epoch must be below num-epochs')
    data = str(Path(args.datapath).expanduser().resolve())
    output = args.output_root.expanduser().resolve() / f'seed_{args.seed}'
    pretrain = args.pretrain
    if pretrain.lower() not in ('none', 'null'):
        pretrain = str(Path(pretrain).expanduser().resolve())
        if args.stage != 'test' and not args.dry_run and not Path(pretrain).is_file():
            parser.error(f'The pre-trained weights do not exist.')
    train = [sys.executable, str(PROJECT/'train_Compare.py'),
             '--datapath', data, '--pretrain', pretrain, '--savepath', str(output),
             '--seed', str(args.seed), '--num_epochs', str(args.num_epochs),
             '--batch_size', str(args.batch_size), '--num-workers', str(args.num_workers),
             '--lr', str(args.lr), '--weight_decay', str(args.weight_decay),
             '--slab-axis', str(args.slab_axis), '--slab-thickness', '26',
             '--slab-overlap', '13', '--extra-width', '12',
             '--validate-start-epoch', str(args.validate_start_epoch)]
    if not args.tensorboard:
        train.append('--no-tensorboard')
    test = [sys.executable, str(PROJECT/'test.py'), 'model_earlystop.pth',
            '--datapath', data, '--checkpoint-dir', str(output), '--output-dir', str(output)]
    commands = ([train] if args.stage == 'train' else [test] if args.stage == 'test' else [train, test])
    print(f'Excel: {PROJECT / "excel"}', flush=True)
    print(f'One experiment, seed={args.seed}; output={output}', flush=True)
    if args.dry_run:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return
    if not Path(data).is_dir():
        parser.error(f'The data directory does not exist：{data}')
    for filename in ('7.xlsx','1.xlsx','2.xlsx'):
        if not (PROJECT/'excel'/filename).is_file():
            parser.error(f'Missing project Excel: {PROJECT / "excel" / filename}')
    if args.stage != 'test' and output.exists() and any(output.iterdir()):
        parser.error(f'Existing training results：{output}；Please set up another one --output-root')
    if args.stage == 'test' and not (output/'model_earlystop.pth').is_file():
        parser.error(f'The weight does not exist：{output / "model_earlystop.pth"}')
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f'Subroutine not completed ( {error.returncode})，Please review the specific errors listed above: ', file=sys.stderr)
        sys.exit(error.returncode)
