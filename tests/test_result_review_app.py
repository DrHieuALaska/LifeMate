"""Dataset-independent checks for the local four-panel review application."""
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave
import zipfile
from http.server import ThreadingHTTPServer

from tools.review_app.public_review import PublicReview
from tools.review_app.dataset_source import DatasetSource
from tools.review_app.open_review import choose_dataset, ensure_results, main as launch_review, results_ready, running_app
from tools.review_app.server import ROOT, Review, make_handler


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class ResultReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='lifemate_review_synthetic_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ('label.html', 'label.css', 'label.js', 'index.html', 'style.css', 'app.js'):
            shutil.copyfile(ROOT / name, self.root / name)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.dataset = self.root / 'dataset.zip'
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\0\0' * 160)
        wav_bytes = buffer.getvalue()
        context = dict(current_time='2030-01-01T09:00:00+07:00', timezone='Asia/Ho_Chi_Minh',
                       user_profile={'name': 'Synthetic User', 'work_address': 'Example Office'},
                       installed_apps={'demo': {'tools': ['demo.lookup']}})
        records = [dict(id='public_demo_001', audio='audio/one.wav', context=context,
                        history=[dict(role='user', audio='audio/history.wav'),
                                 dict(role='assistant', text='Synthetic assistant response', tool_calls=[])]),
                   dict(id='public_demo_002', audio='audio/two.wav', context=context, history=[])]
        catalog = dict(apps=[dict(id='demo', tools=[dict(name='demo.lookup', kind='read',
            description='Synthetic lookup', parameters={'type': 'object', 'properties': {}})])])
        with zipfile.ZipFile(self.dataset, 'w') as archive:
            prefix = 'synthetic/public_test/'
            archive.writestr(prefix + 'public_test.jsonl', ''.join(json.dumps(r) + '\n' for r in records))
            archive.writestr(prefix + 'public_test_tools_catalog.json', json.dumps(catalog))
            for name in ('one.wav', 'two.wav', 'history.wav'):
                archive.writestr(prefix + 'audio/' + name, wav_bytes)
            for r in records:
                archive.writestr(prefix + 'public_test_environment/' + r['id'] + '/state.json',
                                 json.dumps({'demo': {'items': [{'id': 'synthetic_item', 'name': 'Example'}]}}))
            archive.writestr('synthetic/train/audio/valid.wav', wav_bytes)
            archive.writestr('synthetic/train/train.jsonl', json.dumps(dict(id='train_demo',
                audio='audio/valid.wav', context=context, history=[], label={'tool_calls': []})) + '\n')
        ids = [r['id'] for r in records]
        write_json(self.results / 'selected_ids.json', ids)
        asr = [dict(id=uid, audio_text='Synthetic current ASR ' + str(i),
                    history_audio_text=['Synthetic history ASR'] if i == 0 else []) for i, uid in enumerate(ids)]
        predictions = [dict(id=uid, tool_calls=[dict(name='demo.lookup', arguments={})], state_hash='0' * 64) for uid in ids]
        write_jsonl(self.results / 'batch_asr_transcripts.jsonl', asr)
        write_jsonl(self.results / 'BENCHMARK_NOT_FOR_SUBMISSION.jsonl', predictions)
        for pred in predictions:
            write_json(self.results / 'metrics/journal' / (pred['id'] + '.json'),
                       dict(submission=pred, metrics=dict(status='ok', total_seconds=1)))
        write_json(self.root / 'config.json', dict(dataset_zip='dataset.zip', public_results_dir='results',
                                                  annotation_path='annotations/labels.json'))

    def test_public_only_checkout_and_full_input(self):
        store = Review(root=self.root)
        self.assertEqual(store.dataset, self.dataset.resolve())
        self.assertEqual(store.samples, {})  # A private valid snapshot is optional.
        self.assertEqual(store.public.index()['n'], 2)
        full = store.public.full_input('public_demo_001')
        self.assertEqual(full['agent_record_after_asr']['history'][0]['audio_text'], 'Synthetic history ASR')
        self.assertNotIn('audio_text', full['original_manifest_record']['history'][0])
        self.assertEqual(full['initial_environment_state']['demo']['items'][0]['id'], 'synthetic_item')
        self.assertEqual(full['allowed_tool_schemas'][0]['name'], 'demo.lookup')
        # Config-free invocation with explicit paths works in another checkout.
        (self.root / 'config.json').unlink()
        other = Review(root=self.root, dataset=self.dataset, public_results=self.results,
                       annotation_path=self.root / 'other_labels.json')
        self.assertEqual(other.public.ids, store.public.ids)

    def test_optional_valid_snapshot(self):
        snapshot = dict(models=[{'key': 'demo'}], source={}, catalog={'apps': []},
            audio_members={'valid_demo': 'synthetic/train/audio/valid.wav'},
            conversations={'train_demo': dict(id='train_demo', context={'installed_apps': {}},
                                               history=[], label={'tool_calls': []})},
            samples=[dict(uid='valid_demo', conversation_id='train_demo', kind='final', history_index=None,
                          reference='Synthetic reference', duration_s=0.01,
                          predictions={'demo': {'edits': 0, 'text': 'Synthetic reference'}})])
        path = self.root / 'valid.json'
        write_json(path, snapshot)
        store = Review(root=self.root, valid_snapshot=path)
        self.assertEqual(store.detail('valid_demo')['output'], {'tool_calls': []})
        self.assertEqual(store.index[0]['conversation_no'], 1)
        self.assertIsNotNone(store.public)

    def test_extracted_package_and_direct_public_directory(self):
        extracted = self.root / 'extracted'
        # Extract only this test's trusted, generated synthetic ZIP.
        with zipfile.ZipFile(self.dataset) as archive:
            archive.extractall(extracted)
        expected = Review(root=self.root).public.full_input('public_demo_001')
        for directory in (extracted, extracted / 'synthetic', extracted / 'synthetic/public_test'):
            with self.subTest(directory=directory.name):
                store = Review(root=self.root, dataset=directory)
                actual = store.public.full_input('public_demo_001')
                self.assertEqual({k: v for k, v in actual.items() if k != 'source'},
                                 {k: v for k, v in expected.items() if k != 'source'})
                self.assertEqual(store.public.fingerprint, Review(root=self.root).public.fingerprint)
                self.assertAlmostEqual(store.public.detail('public_demo_001')['duration_s'], 0.01)
                self.assertEqual(store.dataset_source.read(store.public.audio_members['public_demo_001'])[:4], b'RIFF')
        reader = DatasetSource(extracted / 'synthetic')
        self.assertEqual(reader.read('old-wrapper/train/audio/valid.wav')[:4], b'RIFF')
        with self.assertRaises(ValueError):
            reader.read('../config.json')
        with self.assertRaises(ValueError):
            reader.read('C:/outside.wav')
        config = json.loads((self.root / 'config.json').read_text(encoding='utf-8'))
        config.update(dataset_path='extracted/synthetic', dataset_zip='missing-legacy.zip')
        write_json(self.root / 'config.json', config)
        self.assertEqual(Review(root=self.root).dataset, (extracted / 'synthetic').resolve())

    def test_extracted_audio_http_seek(self):
        extracted = self.root / 'unpacked'
        with zipfile.ZipFile(self.dataset) as archive:
            archive.extractall(extracted)
        store = Review(root=self.root, dataset=extracted)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(f'http://127.0.0.1:{server.server_port}/public-audio/public_demo_001',
                              headers={'Range': 'bytes=0-43'})
            with urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read()[:4], b'RIFF')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_one_click_sibling_dataset_and_config_override(self):
        app = self.root / 'LifeMate/tools/review_app'
        sibling = self.root / 'train+public_test'
        sibling.mkdir()
        self.assertEqual(choose_dataset(app, {}), sibling.resolve())
        self.assertEqual(choose_dataset(app, {'dataset_zip': str(self.dataset)}), self.dataset.resolve())
        self.assertEqual(choose_dataset(app, {'dataset_zip': 'missing'}, explicit=sibling), sibling.resolve())

    def test_one_click_downloads_only_missing_results(self):
        app = self.root / 'launcher_app'
        path = app / 'imports/kaggle_20_sample_public_test/lifemate_benchmark_634'
        def fake_download(*args, **kwargs):
            shutil.copytree(self.results, path)
        downloader = Mock(side_effect=fake_download)
        self.assertFalse(results_ready(path))
        ensure_results(app, path, runner=downloader)
        self.assertTrue(results_ready(path))
        ensure_results(app, path, runner=downloader)
        downloader.assert_called_once()

    def test_one_click_reopens_matching_server(self):
        info = dict(dataset=str(self.dataset), public_results=str(self.results),
                    annotations=str(self.root / 'annotations/labels.json'))
        with patch('tools.review_app.open_review.running_app', return_value=info), \
             patch('tools.review_app.open_review.webbrowser.open') as browser, \
             patch('tools.review_app.open_review.subprocess.call') as runner, \
             patch('builtins.print'):
            self.assertEqual(launch_review([], root=self.root), 0)
        browser.assert_called_once_with('http://127.0.0.1:8770/label')
        runner.assert_not_called()

    def test_http_audio_annotations_and_export(self):
        store = Review(root=self.root)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            self.assertEqual(running_app(self.root, server.server_port)['application'], 'lifemate-review-app')
            with urlopen(base + '/') as response:
                self.assertIn(b'panel-grid', response.read())
            with urlopen(base + '/api/label/index') as response:
                self.assertFalse(json.load(response)['valid_available'])
            with urlopen(base + '/api/label/input/public_demo_001') as response:
                self.assertEqual(json.load(response)['id'], 'public_demo_001')
            with urlopen(Request(base + '/public-audio/public_demo_001', headers={'Range': 'bytes=0-43'})) as response:
                self.assertEqual(response.status, 206)
                header = response.read()
                self.assertEqual(len(header), 44)
                self.assertEqual(header[:4], b'RIFF')
            note = dict(corrected_transcript='Synthetic corrected text', reviewed=True,
                        asr_judgement='acceptable', reasoning_judgement='correct', expected_output={'tool_calls': []})
            body = json.dumps(dict(source_fingerprint=store.public.fingerprint, annotation=note)).encode()
            url = base + '/api/label/note/public_demo_001'
            with urlopen(Request(url, body, headers={'Content-Type': 'application/json'})) as response:
                self.assertTrue(json.load(response)['annotation']['reviewed'])
            restored = PublicReview(self.root, self.dataset, self.root / 'annotations/labels.json', self.results)
            self.assertEqual(restored.index()['reviewed'], 1)
            with urlopen(base + '/api/label/export.jsonl') as response:
                self.assertEqual(json.loads(response.read())['human_annotation']['corrected_transcript'],
                                 note['corrected_transcript'])
            with self.assertRaises(HTTPError) as rejected:
                urlopen(Request(url, body, headers={'Content-Type': 'application/json', 'Origin': 'https://example.invalid'}))
            self.assertEqual(rejected.exception.code, 403)
            with self.assertRaises(HTTPError) as rejected:
                urlopen(base + '/%2e%2e/config.json')
            self.assertEqual(rejected.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
