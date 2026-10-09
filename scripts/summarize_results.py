"""Export evaluator metrics and the unweighted 15-corruption mean."""
import argparse
import csv
from pathlib import Path
import re

CORRUPTIONS = {'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'}
PATTERN = re.compile(r'^([a-z_]+), ([0-9.]+) \+/- ([0-9.]+), ([0-9.]+) \+/- ([0-9.]+), ([0-9.]+) \+/- ([0-9.]+)$')


def summarize(text):
    rows = []
    seen = set()
    for line in text.splitlines():
        match = PATTERN.match(line.strip())
        if not match:
            continue
        name, *values = match.groups()
        if name in seen:
            raise ValueError(f'Duplicate corruption result: {name}')
        seen.add(name)
        rows.append([name, *map(float, values)])
    if not rows:
        raise ValueError('No evaluator result rows found.')
    corruptions = [row for row in rows if row[0] in CORRUPTIONS]
    if {row[0] for row in corruptions} == CORRUPTIONS:
        # Standard deviations across trials are not averaged or relabeled.
        rows.append(['corruption_mean_15', sum(r[1] for r in corruptions) / 15, '',
                     sum(r[3] for r in corruptions) / 15, '',
                     sum(r[5] for r in corruptions) / 15, ''])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = summarize(args.results.read_text(encoding='utf-8'))
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(['corruption', 'miou_mean', 'miou_std', 'mdice_mean', 'mdice_std', 'macc_mean', 'macc_std'])
        writer.writerows(rows)
    print(f'Exported {len(rows)} rows to {args.output}')
    if not any(row[0] == 'corruption_mean_15' for row in rows):
        print('All 15 corruptions are required for the published corruption average; no partial mean was generated.')


if __name__ == '__main__':
    main()
