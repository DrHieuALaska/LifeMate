"""Deliberately trivial no-action smoke baseline. Reads IDs only; no model or labels."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.evaluation.dataset import read_ids
from lifemate.utils.safety import safe_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='train+public_test/reports/eda/validation_split/dev_ids.txt')
    parser.add_argument('--output', required=True, help='New JSONL file under artifacts/')
    args = parser.parse_args()
    ids = read_ids(args.manifest)
    if len(ids) != 193 or Path(args.manifest).name != 'dev_ids.txt':
        parser.error('Use the existing 193-ID development manifest')
    output = safe_path(args.output)
    if 'artifacts' not in [p.casefold() for p in output.parts]:
        parser.error('Prediction output must be under artifacts/')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        for sid in ids:
            f.write(json.dumps({'id': sid, 'tool_calls': []})+'\n')
    print(f'Wrote {len(ids)} empty predictions. Trivial smoke baseline; no ASR/LLM model was used.')


if __name__ == '__main__':
    main()
