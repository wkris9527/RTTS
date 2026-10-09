"""Report imports, versions, CUDA, and SAM checkpoint without downloading models."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Save the report as JSON.')
    args = parser.parse_args()
    report = {'python': sys.version, 'packages': {}, 'errors': []}
    for name in ['torch', 'torchvision', 'mmcv', 'mmengine', 'mmseg', 'segment_anything',
                 'numpy', 'scipy', 'skimage', 'sklearn', 'timm', 'cv2', 'ftfy', 'regex',
                 'yaml', 'prettytable', 'matplotlib', 'tqdm']:
        try:
            module = importlib.import_module(name)
            report['packages'][name] = getattr(module, '__version__', 'imported')
        except Exception as error:
            report['errors'].append(f'{name}: {type(error).__name__}: {error}')
    try:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['cuda_version'] = torch.version.cuda
        if torch.cuda.is_available():
            report['gpu'] = torch.cuda.get_device_name(0)
        else:
            report['errors'].append('CUDA is unavailable; RTTS requires an NVIDIA GPU.')
    except ImportError:
        pass
    checkpoint = Path(__file__).resolve().parents[1] / 'sam_vit_h_4b8939.pth'
    report['sam_checkpoint_exists'] = checkpoint.is_file()
    if not checkpoint.is_file():
        report['errors'].append('Missing SAM ViT-H checkpoint.')
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
