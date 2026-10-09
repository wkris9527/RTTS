"""Launch the anonymous RTTS experiment without changing its algorithm."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORRUPTIONS = ['original', 'gaussian_noise', 'shot_noise', 'impulse_noise',
               'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
               'snow', 'frost', 'fog', 'brightness', 'contrast',
               'elastic_transform', 'pixelate', 'jpeg_compression']


def build_command(config, args):
    command = [sys.executable, 'main.py', '--adapt', '--method', 'mlmp',
               '--dataset', config['dataset'], '--data_dir', str(Path(args.data_dir).resolve()),
               '--save_dir', str(Path(args.output).resolve()), '--prompt_dir', 'prompts.yaml',
               '--ovss_type', 'naclip', '--ovss_backbone', 'ViT-L/14',
               '--vision_outputs', *map(str, range(-1, -19, -1)), '--alpha_cls', '1.0',
               '--batch-size', '1', '--workers', str(args.workers),
               '--lr', '0.001', '--steps', '10',
               '--trials', str(args.trials), '--seed', str(args.seed),
               '--init_resize', *map(str, config['init_resize']),
               '--patch_size', '224', '224', '--patch_stride', '112',
               '--corruptions_list', *(CORRUPTIONS if args.corruptions == 'all' else [args.corruptions]),
               '--plot_loss', '--class_extensions']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['v20', 'v21', 'p59', 'p60', 'cityscapes', 'coco_obj', 'coco_stuff'], required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', default='outputs/rtts')
    parser.add_argument('--corruptions', choices=['all', *CORRUPTIONS], default='original')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 0:
        parser.error('trials must be positive; workers must be nonnegative.')
    config = json.loads((ROOT / 'configs' / f'{args.benchmark}.json').read_text())
    command = build_command(config, args)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('CUDA_VISIBLE_DEVICES=' + args.gpu)
    print(shlex.join(command))
    if args.dry_run:
        return
    data = Path(args.data_dir).resolve()
    missing = [str(data / path) for path in config['required_paths'] if not (data / path).exists()]
    if missing:
        parser.error('Missing dataset paths:\n' + '\n'.join(missing))
    checkpoint = ROOT / 'sam_vit_h_4b8939.pth'
    if not checkpoint.is_file():
        parser.error('Missing SAM checkpoint. Run python scripts/download_checkpoints.py first.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    record = {'commit': commit, 'argv': command, 'CUDA_VISIBLE_DEVICES': args.gpu,
              'benchmark': args.benchmark, 'preset': config, 'launcher': vars(args)}
    (output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
