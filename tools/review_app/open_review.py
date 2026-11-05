"""One-click launcher: find sibling data, fetch missing results, open the UI."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def configured_path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def choose_dataset(root, config, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
    elif config.get('dataset_path') or config.get('dataset_zip'):
        path = configured_path(root, config.get('dataset_path') or config['dataset_zip'])
    else:
        sibling = root.parent.parent.parent
        choices = [sibling / 'train+public_test', sibling / 'train+public_test.zip']
        path = next((p for p in choices if p.exists()), choices[0])
    if not path.exists():
        raise FileNotFoundError('Không thấy dữ liệu: ' + str(path) +
                                '\nĐặt train+public_test cạnh thư mục repo LifeMate rồi mở lại.')
    return path.resolve()


def results_ready(path):
    required = ('selected_ids.json', 'batch_asr_transcripts.jsonl', 'BENCHMARK_NOT_FOR_SUBMISSION.jsonl')
    if not all((path / name).is_file() for name in required):
        return False
    try:
        ids = json.loads((path / 'selected_ids.json').read_text(encoding='utf-8'))
        return isinstance(ids, list) and bool(ids) and all(
            isinstance(uid, str) and uid.replace('_', '').replace('-', '').isalnum()
            and (path / 'metrics/journal' / (uid + '.json')).is_file() for uid in ids)
    except (ValueError, OSError):
        return False


def ensure_results(root, path, runner=subprocess.run):
    if results_ready(path):
        return
    default = root / 'imports/kaggle_20_sample_public_test/lifemate_benchmark_634'
    if path.resolve() != default.resolve():
        raise FileNotFoundError('Thư mục output tùy chọn chưa đủ file: ' + str(path))
    print('First run: downloading Kaggle review outputs...', flush=True)
    runner([sys.executable, '-B', '-X', 'utf8', str(root / 'fetch_kaggle_results.py')], check=True)
    if not results_ready(path):
        raise FileNotFoundError('Output Kaggle chưa đủ. Kiểm tra kết nối/quyền truy cập rồi mở lại.')


def running_app(root, port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=2) as response:
            info = json.load(response)
        if info.get('application') == 'lifemate-review-app' and Path(info['app_root']).resolve() == root.resolve():
            return info
    except (OSError, ValueError, KeyError):
        pass
    return None


def main(argv=None, root=ROOT):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--public-results', type=Path)
    parser.add_argument('--valid-snapshot', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--port', type=int)
    args = parser.parse_args(argv)
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    dataset = choose_dataset(root, config, args.dataset)
    results = args.public_results.resolve() if args.public_results else configured_path(
        root, config.get('public_results_dir', 'imports/kaggle_20_sample_public_test/lifemate_benchmark_634'))
    annotations = args.annotations.resolve() if args.annotations else configured_path(
        root, config.get('annotation_path', 'data/manual_public_labels.json'))
    port = args.port or config.get('port', 8770)
    info = running_app(root, port)
    if info:
        if Path(info['dataset']).resolve() != dataset or Path(info['public_results']).resolve() != results or Path(info['annotations']).resolve() != annotations:
            raise RuntimeError('App đang mở với dữ liệu/cấu hình khác. Đóng cửa sổ server cũ rồi mở lại.')
        webbrowser.open(f'http://127.0.0.1:{port}/label')
        print('Đã mở app đang chạy.', flush=True)
        return 0
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.5):
            raise RuntimeError(f'Cổng {port} đang được dùng. Đóng server cũ hoặc chọn --port khác.')
    except OSError:
        pass
    ensure_results(root, results)
    command = [sys.executable, '-B', '-X', 'utf8', str(root / 'server.py'),
               '--dataset', str(dataset), '--public-results', str(results),
               '--annotations', str(annotations), '--port', str(port), '--open']
    if args.valid_snapshot:
        command += ['--valid-snapshot', str(args.valid_snapshot.resolve())]
    print('Dữ liệu:', dataset, flush=True)
    print('Giữ cửa sổ này mở khi dùng app. Nhấn Ctrl+C để dừng.', flush=True)
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
