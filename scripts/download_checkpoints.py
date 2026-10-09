"""Download SAM ViT-H from Meta's official checkpoint host."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'sam_vit_h_4b8939.pth'))
    parser.add_argument('--sha256', help='Optional independently obtained expected SHA-256.')
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise SystemExit(f'File already exists: {destination}; refusing to overwrite.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pth.part')
    checksum = hashlib.sha256()
    size = 0
    with urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
        expected_size = int(response.headers.get('Content-Length', 0))
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            checksum.update(chunk)
            size += len(chunk)
            print(f'\rDownloaded {size / 1024**2:.0f} MiB', end='', flush=True)
    if expected_size and size != expected_size:
        raise SystemExit('\nIncomplete download; .part file retained for inspection.')
    digest = checksum.hexdigest()
    if args.sha256 and digest.lower() != args.sha256.lower():
        raise SystemExit('\nChecksum mismatch; .part file retained, checkpoint not installed.')
    temporary.replace(destination)
    print(f'\nSaved: {destination}\nSHA-256: {digest}')


if __name__ == '__main__':
    main()
