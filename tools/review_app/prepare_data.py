"""Create a read-only review snapshot from existing predictions and official labels."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import unicodedata
import zipfile

MODELS = [
    ('zaic5', 'ZAIC · epoch 5', 'gipformer15_zaic_5epochs', 'epoch_05'),
    ('zaic7', 'ZAIC · thêm 2 vòng', 'gipformer15_zaic5_zaic_only_2rounds', 'round_02'),
    ('vimd_zaic', 'ViMD 2 → ZAIC 2', 'gipformer15_vimd02_to_zaic_2epochs', 'epoch_02'),
    ('mixed', 'ZAIC 5 → Mixed 2', 'gipformer15_zaic5_mixed_2rounds', 'round_02'),
]


def normalize(text):
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in 'PS' else c
                           for c in unicodedata.normalize('NFC', text).lower()).split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = {r['uid']: r for r in rows}
    assert len(rows) == len(indexed), f'Duplicate UID: {path}'
    return indexed


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare(workspace, destination):
    base = workspace / 'notebook/training_results/03_transcript_comparison'
    dataset = workspace / 'train+public_test.zip'
    valid = read_jsonl(base / MODELS[0][2] / 'valid.jsonl')
    assert len(valid) == 504
    assert sum(len(normalize(r['reference']).split()) for r in valid.values()) == 7135
    predictions, models = {}, []
    for key, title, process, stage in MODELS:
        other = read_jsonl(base / process / 'valid.jsonl')
        assert set(other) == set(valid)
        assert all((other[u]['reference'], other[u]['audio_relative'], other[u]['conversation_id'],
                    other[u]['kind'], other[u]['history_index']) ==
                   (r['reference'], r['audio_relative'], r['conversation_id'], r['kind'], r['history_index'])
                   for u, r in valid.items())
        path = base / process / stage / 'predictions.jsonl'
        predictions[key] = read_jsonl(path)
        assert set(predictions[key]) == set(valid)
        metric = json.loads((path.parent / 'metrics.json').read_text(encoding='utf-8'))
        models.append(dict(key=key, title=title, stage=stage, process=process,
                           prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           wer_pct=100 * metric['wer'], cer_pct=100 * metric['cer'],
                           exact=metric['exact'], n=metric['n']))
    samples, conversations, audio_members = [], {}, {}
    with zipfile.ZipFile(dataset) as archive:
        names = set(archive.namelist())
        manifests = [n for n in names if PurePosixPath(n).parts[-2:] == ('train', 'train.jsonl')
                     and '__MACOSX' not in PurePosixPath(n).parts]
        assert len(manifests) == 1
        manifest = archive.read(manifests[0])
        assert hashlib.sha256(manifest).hexdigest() == '24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8'
        prefix = str(PurePosixPath(manifests[0]).parent) + '/'
        originals = {r['id']: r for r in
                     (json.loads(line) for line in manifest.decode('utf-8').splitlines() if line.strip())}
        catalog = json.loads(archive.read(prefix + 'train_tools_catalog.json'))

        def audio_key(relative):
            member = prefix + relative
            assert member in names, member
            key = 'audio_' + hashlib.sha256(member.encode()).hexdigest()[:24]
            audio_members[key] = member
            return key

        for uid, row in valid.items():
            conversation = originals[row['conversation_id']]
            if conversation['id'] not in conversations:
                # Copy exact source fields; attach audio links without modifying the source file.
                record = json.loads(json.dumps(conversation, ensure_ascii=False))
                record['audio_key'] = audio_key(record['audio'])
                for turn in record.get('history', []):
                    if turn.get('audio'):
                        turn['audio_key'] = audio_key(turn['audio'])
                conversations[record['id']] = record
            turn = conversation if row['kind'] == 'final' else conversation['history'][row['history_index']]
            assert turn['audio'] == row['audio_relative'] and turn['audio_text'] == row['reference'], uid
            member = prefix + row['audio_relative']
            assert hashlib.sha256(archive.read(member)).hexdigest() == row['wav_sha256'], uid
            audio_members[uid] = member
            ref = normalize(row['reference'])
            scored = {}
            for key, pred in predictions.items():
                p = pred[uid]
                assert p['status'] == 'ok' and isinstance(p['text'], str)
                hyp = normalize(p['text'])
                edits = distance(ref.split(), hyp.split())
                scored[key] = dict(text=p['text'], exact=ref == hyp, edits=edits,
                                   wer_pct=100 * edits / len(ref.split()))
            samples.append(dict(uid=uid, conversation_id=row['conversation_id'], kind=row['kind'],
                                history_index=row['history_index'], reference=row['reference'],
                                duration_s=row['duration_s'], n_words=len(ref.split()),
                                predictions=scored))
        for model in models:
            key = model['key']
            assert sum(s['predictions'][key]['exact'] for s in samples) == model['exact']
            computed = 100 * sum(s['predictions'][key]['edits'] for s in samples) / 7135
            assert abs(computed - model['wer_pct']) < 1e-7
    write(destination / 'data/review.json', dict(version=1, models=models, samples=samples,
          conversations=conversations, catalog=catalog, audio_members=audio_members,
          source=dict(manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                      split='ZAIC internal valid504', reference_tokens=7135)))
    write(destination / 'config.json', dict(dataset_zip=str(dataset.resolve()), port=8770))
    print(f'Prepared {len(samples)} audio, {len(conversations)} conversations, {len(models)} models')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prepare(args.workspace.resolve(), args.destination.resolve())
