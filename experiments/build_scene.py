"""Generate a candidate free-base G1/table/dynamic-block scene, with asset hashes."""

import argparse
import json
from pathlib import Path

from baseline.common import LOCK, ROOT, checked_checkout
from baseline.grasp import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sonic-repo', type=Path, default=ROOT / 'third_party/sonic')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--block-xy', type=float, nargs=2, default=(.4, -.15))
    args = parser.parse_args()
    checked_checkout(args.sonic_repo, json.loads(LOCK.read_text())['sonic']['commit'])
    metadata = build_scene(args.sonic_repo, args.out, args.block_xy,
                           start_back=0.0, direct_start=True)
    print(f'Candidate scene: {args.out}; metadata: {json.dumps(metadata)}')


if __name__ == '__main__':
    main()
