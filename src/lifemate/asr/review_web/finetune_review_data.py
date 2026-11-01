"""Small, private validation bundles shared by Kaggle export and local review.

Only explicit JSON/JSONL members are read. Checkpoints and embedded code are
never loaded or executed. Audio remains in the user's original dataset ZIP.
"""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import unicodedata
import zipfile

MODELS = {'hynt': 'hynt_finetune.py', 'sherpa_vi': 'sherpa_vi_finetune.py'}
MAX_ZIP_BYTES = 32*1024**2
MAX_UNPACKED_BYTES = 64*1024**2


def normalize(text):
    text = unicodedata.normalize('NFC', str(text)).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in text).split())


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def index_jsonl(payload):
    result = {}
    for line in payload.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row.get('uid')
        if not isinstance(uid, str) or not uid or uid in result:
            raise ValueError('Thiếu hoặc trùng UID trong JSONL')
        result[uid] = row
    return result


def run_directory(raw, trainer):
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'Không thấy {root}. Add Input output continuation tương ứng.')
    if (root/'valid.jsonl').is_file() and (root/trainer).is_file():
        return root
    matches = sorted({p.parent for p in root.rglob('valid.jsonl')
                      if (p.parent/trainer).is_file() and (p.parent/'baseline_metrics.json').is_file()})
    if len(matches) != 1:
        raise ValueError(f'Cần chọn một run dưới {root}; tìm thấy {len(matches)}: {matches}')
    return matches[0]


def best_metric(root):
    candidates = []
    for path in root.glob('*_metrics.json'):
        if not (path.name == 'baseline_metrics.json' or path.name.startswith(('epoch_', 'partial_epoch_'))):
            continue
        item = json.loads(path.read_text(encoding='utf-8'))
        b = item['buckets']['all']
        if not b.get('n') or b.get('n_ok') != b['n'] or b.get('n_error') != 0:
            continue
        candidates.append((b['exact_normalized']/b['n'], -b['wer'],
                           -item.get('global_step', 0), path))
    if not candidates:
        raise ValueError(f'Không có metrics validation hoàn chỉnh trong {root}')
    return max(candidates, key=lambda x: x[:3])[3]


def export_bundle(inputs, target, expected_count=504):
    if set(inputs) != set(MODELS):
        raise ValueError('Cần đúng hai input hynt và sherpa_vi')
    members = {}
    selection = {}
    for label, trainer in MODELS.items():
        root = run_directory(inputs[label], trainer)
        metric = best_metric(root)
        predictions = metric.with_name(metric.name.replace('_metrics.json', '_predictions.jsonl'))
        files = {'valid.jsonl': root/'valid.jsonl', 'split_summary.json': root/'split_summary.json',
                 'run_identity.json': root/'run_identity.json', 'metrics.json': metric,
                 'predictions.jsonl': predictions}
        for destination, source in files.items():
            if not source.is_file():
                raise FileNotFoundError(f'Thiếu {source.name} trong {root}. Dùng output đầy đủ, không dùng ZIP handoff chỉ có config.')
            if source.stat().st_size > MAX_ZIP_BYTES:
                raise ValueError(f'File bất thường quá lớn: {source}')
            members[label+'/'+destination] = source.read_bytes()
        selection[label] = dict(source_directory=str(root), metrics_file=metric.name,
                                predictions_file=predictions.name, weights_verified=False)
    manifest = dict(format='lifemate-valid-review', version=1, selections=selection,
                    expected_count=expected_count, files=[dict(path=k, sha256=hashlib.sha256(v).hexdigest())
                                                        for k, v in members.items()])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr('review_manifest.json', json_bytes(manifest))
    payload = buffer.getvalue()
    # Recompute scores, references, set membership and split before delivering.
    read_bundle(payload, expected_count=expected_count)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_bytes(payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_bundle(payload, expected_count=504):
    if len(payload) > MAX_ZIP_BYTES:
        raise ValueError('ZIP vượt 32 MiB; cần ZIP review nhỏ, không chứa checkpoint/audio')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or len(names) > 40:
            raise ValueError('ZIP có member trùng hoặc quá nhiều file')
        if sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
            raise ValueError('ZIP giải nén vượt 64 MiB')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Tên member ZIP không hợp lệ')
        if 'review_manifest.json' not in names:
            raise ValueError('Đây chưa phải ZIP review. Chạy notebook export_valid_review; ZIP handoff không chứa predictions.')
        manifest = json.loads(archive.read('review_manifest.json'))
        if manifest.get('format') != 'lifemate-valid-review' or manifest.get('version') != 1:
            raise ValueError('Sai định dạng ZIP review')
        expected_names = {label+'/'+name for label in MODELS for name in
                          ('valid.jsonl', 'split_summary.json', 'run_identity.json', 'metrics.json', 'predictions.jsonl')}
        if set(names) != expected_names | {'review_manifest.json'}:
            raise ValueError('ZIP review có file thiếu/thừa')
        hashes = {entry['path']: entry['sha256'] for entry in manifest['files']}
        if set(hashes) != expected_names or len(hashes) != len(manifest['files']):
            raise ValueError('Danh sách checksum không hợp lệ')
        content = {n: archive.read(n) for n in expected_names}
        for name, value in content.items():
            if hashlib.sha256(value).hexdigest() != hashes[name]:
                raise ValueError('Checksum không khớp: '+name)
    datasets = {}
    models = {}
    fingerprint = None
    for label in MODELS:
        valid = index_jsonl(content[label+'/valid.jsonl'])
        predictions = index_jsonl(content[label+'/predictions.jsonl'])
        split = json.loads(content[label+'/split_summary.json'])
        identity = json.loads(content[label+'/run_identity.json'])
        metrics = json.loads(content[label+'/metrics.json'])
        fp = split.get('split_fingerprint')
        if not fp or any(x.get('split_fingerprint') != fp for x in (identity, metrics)):
            raise ValueError(label+': split fingerprint không khớp')
        if fingerprint is not None and fingerprint != fp:
            raise ValueError('Hai model khác split, không ghép so sánh')
        fingerprint = fp
        if not valid or len(valid) != split.get('valid_audio') or len(valid) != expected_count:
            raise ValueError(f'{label}: cần đủ {expected_count} valid audio, có {len(valid)}')
        if set(valid) != set(predictions):
            raise ValueError(label+': predictions thiếu/thừa UID so với valid')
        scored = {}
        for uid, gold in valid.items():
            pred = predictions[uid]
            if pred.get('status') != 'ok' or not isinstance(pred.get('hypothesis'), str):
                raise ValueError(f'{label}: prediction lỗi/thiếu tại {uid}; không bỏ qua mẫu lỗi')
            if normalize(pred.get('reference')) != normalize(gold['reference']):
                raise ValueError(label+': reference khác valid tại '+uid)
            ref, hyp = normalize(gold['reference']), normalize(pred['hypothesis'])
            scored[uid] = dict(text=pred['hypothesis'], exact=ref == hyp,
                               word_edits=distance(ref.split(), hyp.split()), n_words=len(ref.split()),
                               char_edits=distance(ref.replace(' ', ''), hyp.replace(' ', '')),
                               n_chars=len(ref.replace(' ', '')))
        totals = {key: sum(row[key] for row in scored.values())
                  for key in ('word_edits', 'n_words', 'char_edits', 'n_chars')}
        exact = sum(row['exact'] for row in scored.values())
        saved = metrics['buckets']['all']
        for key, value in dict(totals, n=len(valid), n_ok=len(valid), n_error=0, exact_normalized=exact).items():
            if saved.get(key) != value:
                raise ValueError(f'{label}: metrics không khớp predictions ({key})')
        if totals['n_words'] <= 0 or totals['n_chars'] <= 0:
            raise ValueError('Reference không có từ/ký tự để tính điểm')
        wer, cer = totals['word_edits']/totals['n_words'], totals['char_edits']/totals['n_chars']
        if not math.isclose(wer, saved['wer'], abs_tol=1e-9) or not math.isclose(cer, saved['cer'], abs_tol=1e-9):
            raise ValueError(label+': WER/CER lưu sẵn không khớp tính lại')
        models[label] = dict(epoch=metrics.get('epoch'), global_step=metrics.get('global_step'),
                            exact=exact, n=len(valid), em=exact/len(valid), wer=wer, cer=cer,
                            run_identity=identity.get('identity'),
                            selection=manifest['selections'][label], predictions=scored)
        datasets[label] = valid
    valid = datasets['hynt']
    other = datasets['sherpa_vi']
    if set(valid) != set(other):
        raise ValueError('Hai model không cùng tập valid UID')
    rows = []
    for uid, gold in valid.items():
        for key in ('reference', 'conversation_id', 'kind', 'audio_relative', 'wav_sha256', 'pcm_sha256'):
            if gold.get(key) != other[uid].get(key):
                raise ValueError(f'UID {uid} khác {key} giữa hai run')
        relative = PurePosixPath(gold['audio_relative'])
        if (relative.is_absolute() or '..' in relative.parts or '\\' in str(relative)
                or ':' in str(relative) or relative.parts[0] != 'audio' or relative.suffix.lower() != '.wav'):
            raise ValueError('Audio phải là đường dẫn train/audio WAV tương đối')
        rows.append(dict(uid=uid, conversation_id=gold['conversation_id'], kind=gold['kind'],
                         reference=gold['reference'], audio_relative=str(relative),
                         wav_sha256=gold.get('wav_sha256'), duration_s=gold.get('duration_s', 0),
                         english_term_hint=bool(gold.get('english_term_hint')),
                         contains_digits=any(c.isdigit() for c in gold['reference']),
                         predictions={m: models[m]['predictions'][uid] for m in MODELS}))
    for model in models.values():
        model.pop('predictions')
    # Notes belong to the same contents even if the ZIP is exported again with
    # different container timestamps/compression.
    bundle_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode('utf-8')).hexdigest()
    return dict(bundle_id=bundle_id, split_fingerprint=fingerprint,
                models=models, rows=rows, independent_test=False)
