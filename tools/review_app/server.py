"""Local, read-only LifeMate results viewer. Python standard library only."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlparse, unquote
import webbrowser
try:
    from .public_review import PublicReview
    from .dataset_source import DatasetSource
except ImportError:
    from public_review import PublicReview
    from dataset_source import DatasetSource

ROOT = Path(__file__).resolve().parent


class Review:
    def __init__(self, root=ROOT, dataset=None, public_results=None, valid_snapshot=None, annotation_path=None):
        self.root = Path(root).resolve()
        config_path = self.root / 'config.json'
        self.config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}

        def local_path(argument, key, default=None):
            if argument is not None:
                return Path(argument).expanduser().resolve()
            raw = self.config.get(key, default)
            if raw is None:
                return None
            path = Path(raw).expanduser()
            return path.resolve() if path.is_absolute() else (self.root / path).resolve()

        snapshot = local_path(valid_snapshot, 'valid_snapshot', 'data/review.json')
        self.data = json.loads(snapshot.read_text(encoding='utf-8')) if snapshot.is_file() else dict(
            samples=[], conversations={}, models=[], catalog={'apps': []}, audio_members={}, source={})
        self.samples = {r['uid']: r for r in self.data['samples']}
        self.conversation_numbers = {cid: i + 1 for i, cid in enumerate(sorted(self.data['conversations']))}
        self.dataset = local_path(dataset, 'dataset_path' if 'dataset_path' in self.config else 'dataset_zip')
        if self.dataset is None or not self.dataset.exists():
            raise FileNotFoundError('Set --dataset <ZIP or extracted directory>, or dataset_path in local config.json.')
        self.dataset_source = DatasetSource(self.dataset)
        self.tools = {t['name']: t for app in self.data['catalog']['apps'] for t in app['tools']}
        public_source = local_path(public_results, 'public_results_dir', 'imports/kaggle_20_sample_public_test/lifemate_benchmark_634')
        if (public_results is not None or 'public_results_dir' in self.config) and not (public_source / 'selected_ids.json').is_file():
            raise FileNotFoundError('Public results folder needs selected_ids.json; see tools/review_app/README.md.')
        self.public = PublicReview(self.root, self.dataset,
            local_path(annotation_path, 'annotation_path', 'data/manual_public_labels.json'), public_source
        ) if (public_source / 'selected_ids.json').is_file() else None
        if not self.samples and self.public is None:
            raise FileNotFoundError('Add Kaggle outputs with fetch_kaggle_results.py / --public-results, or supply --valid-snapshot.')
        self.index = []
        ordered = sorted(self.samples.values(), key=lambda row: (
            self.conversation_numbers[row['conversation_id']],
            row['kind'] == 'final', row['history_index'] if row['history_index'] is not None else -1))
        for row in ordered:
            c = self.data['conversations'][row['conversation_id']]
            output, _ = self.output_for(row, c)
            self.index.append(dict(uid=row['uid'], conversation_id=row['conversation_id'], kind=row['kind'],
                **self.numbering(row),
                reference=row['reference'], duration_s=row['duration_s'],
                errors={k: v['edits'] for k, v in row['predictions'].items()},
                search_text=' '.join([row['uid'], row['conversation_id'], row['reference'],
                    *[p['text'] for p in row['predictions'].values()],
                    *[t.get('name', '') for t in output.get('tool_calls', [])]]).lower()))

    def numbering(self, row):
        c = self.data['conversations'][row['conversation_id']]
        user_indices = [i for i, h in enumerate(c.get('history', [])) if h.get('role') == 'user']
        turn_no = len(user_indices) + 1 if row['kind'] == 'final' else user_indices.index(row['history_index']) + 1
        return dict(conversation_no=self.conversation_numbers[row['conversation_id']],
                    conversation_count=len(self.conversation_numbers), turn_no=turn_no,
                    turn_count=len(user_indices) + 1)

    @staticmethod
    def output_for(row, conversation):
        if row['kind'] == 'final':
            return conversation.get('label', {}), 'Nhãn chuẩn của yêu cầu cuối (label)'
        history = conversation.get('history', [])
        index = row['history_index']
        response = history[index + 1] if index + 1 < len(history) else {}
        if response.get('role') != 'assistant':
            return {}, 'Không có phản hồi assistant ngay sau lượt này'
        return {k: v for k, v in response.items() if k not in ('role', 'audio_key')}, \
            f'Phản hồi chuẩn trong history[{index + 1}]'

    def detail(self, uid):
        row = self.samples[uid]
        c = self.data['conversations'][row['conversation_id']]
        output, source = self.output_for(row, c)
        # installed_apps is the source of availability, not the global catalog or gold calls.
        groups = []
        for app, entry in c['context'].get('installed_apps', {}).items():
            groups.append(dict(app=app, tools=[dict(name=name, **{
                k: v for k, v in self.tools.get(name, {}).items() if k != 'name'
            }) for name in entry.get('tools', [])]))
        cut = len(c.get('history', [])) if row['kind'] == 'final' else row['history_index']
        allowed = {t['name'] for group in groups for t in group['tools']}
        return dict(**row, **self.numbering(row), output=output, output_source=source, context=c['context'],
                    history_before=c.get('history', [])[:cut], allowed_tool_groups=groups,
                    gold_tools_not_listed=[t['name'] for t in output.get('tool_calls', [])
                                           if t['name'] not in allowed],
                    reasoning_prediction=None)


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

        def send(self, status, body=b'', content_type='application/json; charset=utf-8', headers=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def json(self, value):
            self.send(200, json.dumps(value, ensure_ascii=False).encode('utf-8'))

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            host = self.headers.get('Host', '')
            if host.split(':')[0] not in ('127.0.0.1', 'localhost'):
                self.send(403, b'{"error":"Local access only"}')
                return
            if self.headers.get('Origin') not in (None, 'http://' + host):
                self.send(403, b'{"error":"Cross-origin write rejected"}')
                return
            path = urlparse(self.path).path
            if not path.startswith('/api/label/note/') or not store.public:
                self.send(404, b'{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 256 * 1024 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Expected a JSON annotation smaller than 256 KB')
                uid = path.removeprefix('/api/label/note/')
                note = store.public.save(uid, json.loads(self.rfile.read(length)))
                self.json(dict(annotation=note))
            except (ValueError, TypeError) as error:
                self.send(400, json.dumps(dict(error=str(error)), ensure_ascii=False).encode('utf-8'))
            except OSError:
                self.send(500, b'{"error":"Cannot save annotations on disk"}')

        def do_GET(self):
            # Only the local hostnames are supported. No cross-origin data API.
            host = self.headers.get('Host', '').split(':')[0]
            if host not in ('127.0.0.1', 'localhost'):
                self.send(403, b'{"error":"Local access only"}')
                return
            path = unquote(urlparse(self.path).path)
            if path == '/api/health':
                self.json(dict(application='lifemate-review-app', app_root=str(store.root),
                    dataset=str(store.dataset), public_results=str(store.public.source) if store.public else '',
                    annotations=str(store.public.annotation_path) if store.public else ''))
            elif path == '/api/label/index' and store.public:
                self.json(dict(store.public.index(), valid_available=bool(store.samples)))
            elif path == '/api/label/export.jsonl' and store.public:
                self.send(200, store.public.export(), 'application/x-ndjson; charset=utf-8',
                          {'Content-Disposition': 'attachment; filename="lifemate_public_human_labels.jsonl"'})
            elif path.startswith('/api/label/sample/') and store.public:
                uid = path.removeprefix('/api/label/sample/')
                if uid not in store.public.rows:
                    self.send(404, b'{"error":"Unknown sample"}')
                else:
                    self.json(store.public.detail(uid))
            elif path.startswith('/api/label/environment/') and store.public:
                uid = path.removeprefix('/api/label/environment/')
                if uid not in store.public.rows:
                    self.send(404, b'{"error":"Unknown sample"}')
                else:
                    self.json(store.public.environment(uid))
            elif path.startswith('/api/label/input/') and store.public:
                uid = path.removeprefix('/api/label/input/')
                if uid not in store.public.rows:
                    self.send(404, b'{"error":"Unknown sample"}')
                else:
                    self.json(store.public.full_input(uid))
            elif path == '/api/index':
                self.json(dict(samples=store.index, models=store.data['models'], source=store.data['source']))
            elif path.startswith('/api/sample/'):
                uid = path.removeprefix('/api/sample/')
                if uid not in store.samples:
                    self.send(404, b'{"error":"Unknown sample"}')
                else:
                    self.json(store.detail(uid))
            elif path.startswith('/audio/') or path.startswith('/public-audio/'):
                public_audio = path.startswith('/public-audio/')
                key = path.removeprefix('/public-audio/' if public_audio else '/audio/')
                member = (store.public.audio_members.get(key) if store.public else None) if public_audio else store.data['audio_members'].get(key)
                if not member:
                    self.send(404, b'{"error":"Unknown audio"}')
                    return
                try:
                    payload = store.dataset_source.read(member)
                except (OSError, ValueError):
                    self.send(404, b'{"error":"Audio not found in the selected dataset source"}')
                    return
                size = len(payload)
                start, end, status = 0, size - 1, 200
                range_header = self.headers.get('Range')
                if range_header:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
                    if not match or not any(match.groups()):
                        self.send(416, b'', headers={'Content-Range': f'bytes */{size}'})
                        return
                    first, last = match.groups()
                    if first:
                        start, end = int(first), min(int(last) if last else size - 1, size - 1)
                    else:
                        start = max(0, size - int(last))
                    if start >= size or start > end:
                        self.send(416, b'', headers={'Content-Range': f'bytes */{size}'})
                        return
                    status = 206
                headers = {'Accept-Ranges': 'bytes'}
                if status == 206:
                    headers['Content-Range'] = f'bytes {start}-{end}/{size}'
                self.send(status, payload[start:end + 1], 'audio/wav', headers)
            else:
                if path == '/valid' and not store.samples:
                    self.send(404, b'{"error":"No local valid snapshot; use --valid-snapshot <review.json>."}')
                    return
                assets = {'/': ('label.html' if store.public else 'index.html', 'text/html; charset=utf-8'),
                          '/label': ('label.html', 'text/html; charset=utf-8'),
                          '/valid': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                          '/label.js': ('label.js', 'text/javascript; charset=utf-8'),
                          '/label.css': ('label.css', 'text/css; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path not in assets:
                    self.send(404, b'{"error":"Not found"}')
                    return
                file, mime = assets[path]
                self.send(200, (store.root / file).read_bytes(), mime)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--dataset', type=Path, help='Original ZIP, extracted package directory, or public_test directory')
    parser.add_argument('--public-results', type=Path, help='Kaggle result folder containing selected_ids.json')
    parser.add_argument('--valid-snapshot', type=Path, help='Optional private valid comparison snapshot')
    parser.add_argument('--annotations', type=Path, help='Optional local annotation JSON path')
    args = parser.parse_args()
    store = Review(dataset=args.dataset, public_results=args.public_results,
                   valid_snapshot=args.valid_snapshot, annotation_path=args.annotations)
    port = args.port or store.config.get('port', 8770)
    with ThreadingHTTPServer(('127.0.0.1', port), make_handler(store)) as server:
        description = f'{len(store.public.ids)} public conversations for human labeling' if store.public else f'{len(store.samples)} valid audio | 4 models'
        print(f'LifeMate Review: http://127.0.0.1:{port}/ | {description}', flush=True)
        if args.open:
            webbrowser.open(f'http://127.0.0.1:{port}/')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
