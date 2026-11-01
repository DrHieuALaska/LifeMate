"""Local-only two-model audio review. Python stdlib; no GPU/model required."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import sys
from urllib.parse import urlparse
import zipfile

try:
    from .finetune_review_data import read_bundle, json_bytes, MAX_ZIP_BYTES
except ImportError:
    from finetune_review_data import read_bundle, json_bytes, MAX_ZIP_BYTES

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT/'review_web'
CHOICES = {
    'reference_status': {'', 'correct', 'incorrect', 'unsure'},
    'hynt_judgement': {'', 'correct', 'same_meaning', 'changed_meaning', 'unsure'},
    'sherpa_vi_judgement': {'', 'correct', 'same_meaning', 'changed_meaning', 'unsure'},
}
ISSUES = {'name', 'number', 'time', 'negation', 'missing', 'extra', 'english', 'reference', 'noise', 'other'}


def validate_note(note):
    if not isinstance(note, dict) or set(note)-set(CHOICES)-{'listened', 'issues', 'corrected_reference', 'comment', 'updated_at'}:
        raise ValueError('Nhãn ghi chú không hợp lệ')
    result = {}
    for key, allowed in CHOICES.items():
        value = note.get(key, '')
        if not isinstance(value, str) or value not in allowed:
            raise ValueError('Giá trị nhãn không hợp lệ: '+key)
        result[key] = value
    if not isinstance(note.get('listened', False), bool):
        raise ValueError('listened phải là boolean')
    result['listened'] = note.get('listened', False)
    issues = note.get('issues', [])
    if not isinstance(issues, list) or any(not isinstance(x, str) or x not in ISSUES for x in issues):
        raise ValueError('Nhóm lỗi không hợp lệ')
    result['issues'] = sorted(set(issues))
    for key in ('corrected_reference', 'comment'):
        value = note.get(key, '')
        if not isinstance(value, str) or len(value) > 10000:
            raise ValueError('Ghi chú phải là text, tối đa 10.000 ký tự')
        result[key] = value
    result['updated_at'] = datetime.now(timezone.utc).isoformat()
    return result


class ReviewStore:
    def __init__(self, dataset, output, expected_count=504):
        self.dataset, self.output = Path(dataset), Path(output)
        self.expected_count = expected_count
        self.data, self.notes, self.audio_members = None, {}, {}
        self.lock = threading.RLock()

    @staticmethod
    def atomic(path, payload):
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)

    def load(self, payload):
        data = read_bundle(payload, self.expected_count)
        if not self.dataset.is_file():
            raise FileNotFoundError('Thiếu ZIP audio gốc. Khởi động lại với --dataset <đường-dẫn-train+public_test.zip>')
        mapping = {}
        with zipfile.ZipFile(self.dataset) as audio:
            all_names = audio.namelist()
            if len(all_names) != len(set(all_names)):
                raise ValueError('ZIP dataset có tên member trùng')
            names = set(all_names)
            for row in data['rows']:
                found = [prefix+row['audio_relative'] for prefix in ('train+public_test/train/', 'train/')
                         if prefix+row['audio_relative'] in names]
                if len(found) != 1:
                    raise ValueError('Thiếu hoặc trùng audio train: '+row['audio_relative'])
                mapping[row['uid']] = found[0]
        with self.lock:
            self.output.mkdir(parents=True, exist_ok=True)
            note_path = self.output/(data['bundle_id']+'.notes.json')
            notes = {}
            if note_path.is_file():
                existing = json.loads(note_path.read_text(encoding='utf-8'))
                if existing.get('bundle_id') != data['bundle_id']:
                    raise ValueError('File ghi chú thuộc bundle khác')
                for uid, note in existing.get('notes', {}).items():
                    if uid not in mapping:
                        raise ValueError('UID ghi chú không thuộc bundle')
                    notes[uid] = validate_note(note)
            bundle_path = self.output/(data['bundle_id']+'.zip')
            if not bundle_path.exists():
                self.atomic(bundle_path, payload)
            self.data, self.notes, self.audio_members = data, notes, mapping
            self.atomic(self.output/'latest.json', json_bytes({'bundle_id': data['bundle_id']}))
        return self.view()

    def restore_latest(self):
        marker = self.output/'latest.json'
        if marker.is_file():
            key = json.loads(marker.read_text(encoding='utf-8'))['bundle_id']
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('latest.json không hợp lệ')
            self.load((self.output/(key+'.zip')).read_bytes())

    def view(self):
        with self.lock:
            return dict(loaded=bool(self.data), dataset_available=self.dataset.is_file(), data=self.data, notes=self.notes)

    def check(self, bundle_id):
        if not self.data or bundle_id != self.data['bundle_id']:
            raise ValueError('Bundle đã thay đổi hoặc chưa nạp; tải lại trang trước khi lưu')

    def persist(self, notes):
        self.atomic(self.output/(self.data['bundle_id']+'.notes.json'),
                    json_bytes(dict(version=1, bundle_id=self.data['bundle_id'], notes=notes)))
        self.notes = notes

    def save_note(self, bundle_id, uid, note):
        clean = validate_note(note)
        with self.lock:
            self.check(bundle_id)
            if uid not in self.audio_members:
                raise ValueError('UID không thuộc bundle')
            updated = dict(self.notes)
            updated[uid] = clean
            self.persist(updated)
            return clean

    def import_notes(self, backup):
        with self.lock:
            self.check(backup.get('bundle_id'))
            incoming = backup.get('notes')
            if backup.get('version') != 1 or not isinstance(incoming, dict):
                raise ValueError('Sai định dạng backup')
            updated = dict(self.notes)
            for uid, note in incoming.items():
                if uid not in self.audio_members:
                    raise ValueError('Backup có UID lạ; không nhập một phần')
                updated[uid] = validate_note(note)
            self.persist(updated)
            return self.notes

    def backup(self):
        with self.lock:
            if not self.data:
                raise ValueError('Chưa nạp dữ liệu')
            return dict(version=1, bundle_id=self.data['bundle_id'], notes=self.notes)

    def audio(self, uid):
        with self.lock:
            if uid not in self.audio_members:
                raise ValueError('Audio không thuộc tập review')
            member = self.audio_members[uid]
            expected = next(r['wav_sha256'] for r in self.data['rows'] if r['uid'] == uid)
        with zipfile.ZipFile(self.dataset) as archive:
            if archive.getinfo(member).file_size > 32*1024**2:
                raise ValueError('Audio bất thường quá lớn')
            payload = archive.read(member)
        if not expected or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Audio không khớp WAV SHA256 của run; không phát nhầm file')
        return payload

    def csv(self):
        backup = self.backup()
        rows = {r['uid']: r for r in self.data['rows']}
        stream = io.StringIO(newline='')
        fields = ['uid', 'conversation_id', 'kind', 'reference', 'hynt', 'sherpa_vi',
                  'listened', *CHOICES, 'issues', 'corrected_reference', 'comment', 'updated_at']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for uid, note in backup['notes'].items():
            row = rows[uid]
            entry = dict(uid=uid, conversation_id=row['conversation_id'], kind=row['kind'], reference=row['reference'],
                         hynt=row['predictions']['hynt']['text'], sherpa_vi=row['predictions']['sherpa_vi']['text'], **note)
            entry['issues'] = '|'.join(entry['issues'])
            entry = {k: "'"+v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v for k, v in entry.items()}
            writer.writerow(entry)
        return ('\ufeff'+stream.getvalue()).encode('utf-8')


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, kind='application/json; charset=utf-8', download=None):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="'+download+'"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                static = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
                if path in static:
                    name, kind = static[path]
                    self.reply(200, (WEB/name).read_bytes(), kind+'; charset=utf-8')
                elif path == '/api/data':
                    self.reply(200, json_bytes(store.view()))
                elif path.startswith('/audio/'):
                    self.reply(200, store.audio(path[len('/audio/'):]), 'audio/wav')
                elif path == '/api/export.json':
                    self.reply(200, json_bytes(store.backup()), download='lifemate-review-notes.json')
                elif path == '/api/export.csv':
                    self.reply(200, store.csv(), 'text/csv; charset=utf-8', 'lifemate-review-notes.csv')
                else:
                    self.reply(404, json_bytes({'error': 'Không có đường dẫn này'}))
            except (ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile) as exc:
                self.reply(400, json_bytes({'error': str(exc)}))

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and origin != 'http://'+self.headers.get('Host', ''):
                    raise ValueError('Chỉ nhận thao tác từ trang local này')
                path = urlparse(self.path).path
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= (MAX_ZIP_BYTES if path == '/api/load' else 4*1024**2):
                    raise ValueError('Request rỗng hoặc vượt giới hạn')
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('Upload chưa đủ byte')
                if path == '/api/load':
                    result = store.load(body)
                elif path == '/api/note':
                    value = json.loads(body)
                    result = store.save_note(value['bundle_id'], value['uid'], value['note'])
                elif path == '/api/import':
                    result = store.import_notes(json.loads(body))
                else:
                    self.reply(404, json_bytes({'error': 'Không có đường dẫn này'}))
                    return
                self.reply(200, json_bytes(result))
            except (ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile) as exc:
                self.reply(400, json_bytes({'error': str(exc)}))
            except OSError:
                self.reply(500, json_bytes({'error': 'Không ghi được dữ liệu local; kiểm tra quyền và dung lượng'}))
    return Handler


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/train+public_test.zip')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/finetune_review_web')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    store = ReviewStore(args.dataset, args.output)
    if args.bundle:
        store.load(args.bundle.read_bytes())
    else:
        store.restore_latest()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(store))
    print(f'LifeMate review: http://127.0.0.1:{server.server_port}', flush=True)
    print('CPU/local only. Ctrl+C để dừng. Ghi chú:', args.output, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
