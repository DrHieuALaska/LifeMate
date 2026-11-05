"""Fetch official public Kaggle kernel results without exposing download URLs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.request

KERNEL = 'drhieualaska/20-sample-public-test'
ENDPOINT = 'https://api.kaggle.com/v1/kernels.KernelsApiService/ListKernelSessionOutput'
DESTINATION = Path(__file__).resolve().parent / 'imports/kaggle_20_sample_public_test'


def inventory():
    token, seen, files, log = None, set(), [], ''
    while True:
        data = dict(userName=KERNEL.split('/')[0], kernelSlug=KERNEL.split('/')[1], pageSize=100)
        if token:
            data['pageToken'] = token
        request = urllib.request.Request(ENDPOINT, json.dumps(data).encode(),
            headers={'Content-Type': 'application/json', 'User-Agent': 'kaggle-api/v1.7.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
        files.extend(page.get('files', []))
        log = page.get('log') or log
        token = page.get('nextPageToken')
        if not token:
            break
        if token in seen:
            raise ValueError('Repeated Kaggle page token')
        seen.add(token)
    assert len({f['fileName'] for f in files}) == len(files), 'Duplicate output filename'
    return files, log


def fetch(item):
    relative = PurePosixPath(item['fileName'])
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative):
        raise ValueError('Invalid artifact path')
    target = DESTINATION.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        payload = target.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(item['url'], timeout=90) as response:
                    payload = response.read()
                temporary = target.with_suffix(target.suffix + '.part')
                temporary.write_bytes(payload)
                temporary.replace(target)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                if attempt == 2:
                    # Do not expose signed URLs embedded in network exception messages.
                    raise RuntimeError('Download failed: ' + str(relative)) from None
                time.sleep(attempt + 1)
    return dict(file_name=str(relative), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def save_manifest(files):
    # Keep previously downloaded results when fetching only a few additional files.
    cached = []
    for item in files:
        relative = PurePosixPath(item['fileName'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative):
            raise ValueError('Invalid artifact path')
        target = DESTINATION.joinpath(*relative.parts)
        if target.is_file():
            payload = target.read_bytes()
            cached.append(dict(file_name=str(relative), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    (DESTINATION / 'download_manifest.json').write_text(json.dumps(dict(kernel=KERNEL,
        files=sorted(cached, key=lambda f: f['file_name'])), indent=2), encoding='utf-8')
    return len(cached)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--include', action='append', help='Exact file names; default: result data/text files')
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    files, log = inventory()
    safe = [{k: v for k, v in f.items() if k != 'url'} for f in files]
    (DESTINATION / 'output_inventory.json').write_text(json.dumps(dict(kernel=KERNEL, files=safe), indent=2), encoding='utf-8')
    if log:
        (DESTINATION / 'kernel.log').write_text(log, encoding='utf-8')
    print('Inventory:', len(files), 'files', flush=True)
    from collections import Counter
    print('Extensions:', dict(Counter(PurePosixPath(f['fileName']).suffix for f in files)), flush=True)
    for f in files:
        if '/journal/' not in f['fileName']:
            print(f['fileName'], flush=True)
    if args.inventory_only:
        return
    suffixes = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.log', '.html'}
    selected = [f for f in files if f['fileName'] in args.include] if args.include else [
        f for f in files if PurePosixPath(f['fileName']).suffix.lower() in suffixes]
    if args.include and len(selected) != len(set(args.include)):
        raise ValueError('Requested file not present in output')
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, f) for f in selected]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 50 == 0:
                print('Downloaded', len(results), '/', len(selected), flush=True)
    save_manifest(files)
    print('Saved', len(results), 'files;', sum(r['bytes'] for r in results), 'bytes', flush=True)


if __name__ == '__main__':
    main()
