"""Synthetic checks only: no competition data, model downloads or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import ast

import pandas as pd
from lifemate.asr import asr_benchmark as b


class BenchmarkTests(unittest.TestCase):
    def notebook_cells(self):
        p=Path(__file__).resolve().parents[1]/'notebook/lifemate_asr_benchmark.ipynb'
        n=json.loads(p.read_text(encoding='utf-8'))
        return n,{c['metadata']['lifemate_role']:''.join(c['source']) for c in n['cells'] if c['cell_type']=='code'}

    def rows(self):
        return pd.DataFrame([
            dict(ref_uid='a',sample_uid='s1',kind='final',duration=1.,reference='xin chào'),
            dict(ref_uid='b',sample_uid='s2',kind='history',duration=2.,reference='không gửi'),
        ])

    def test_normalization_and_edits(self):
        self.assertEqual(b.normalize('  XIN, CHÀO! '),'xin chào')
        self.assertNotEqual(b.normalize('250000'),b.normalize('hai trăm năm mươi nghìn'))
        self.assertEqual(b.edit_distance(['a','b'],['a','c','d']),2)

    def test_whisper_raw_eos_and_truncated_sequences(self):
        self.assertTrue(b.whisper_sequence_diagnostics([1,10,2],2,2)['ended_with_eos'])
        self.assertTrue(b.whisper_sequence_diagnostics([1,10,2,0,0],[2,3],0)['ended_with_eos'])
        self.assertFalse(b.whisper_sequence_diagnostics([1,10,11],2,2)['ended_with_eos'])
        self.assertFalse(b.whisper_sequence_diagnostics([],2,2)['ended_with_eos'])
        with self.assertRaises(ValueError): b.whisper_sequence_diagnostics([1,10],None)

    def test_transcribe_uses_raw_output_and_still_rejects_real_truncation(self):
        # Exercise ASR.transcribe with synthetic audio/tokens; no torch, weights or GPU required.
        import contextlib
        import types
        from unittest.mock import patch
        import numpy as np
        class Tokens:
            def __init__(self,values): self.values=values
            def __getitem__(self,index): return self.values[index]
        class Features:
            def to(self,*args,**kwargs): return self
        class Processor:
            def __call__(self,*args,**kwargs): return {'input_features':Features()}
            def batch_decode(self,ids,**kwargs): return ['synthetic transcript']
        class Model:
            generation_config=types.SimpleNamespace(eos_token_id=2,pad_token_id=2)
            truncated=False
            def generate(self,**kwargs):
                # Default Whisper tensor output loses EOS. The fixed call must request raw output.
                if not kwargs.get('return_dict_in_generate'): return Tokens([[1,10,11]])
                return types.SimpleNamespace(sequences=Tokens([[1,10,11] if self.truncated else [1,10,11,2]]))
        class Monitor:
            peak=1
            def __enter__(self): return self
            def __exit__(self,*args): pass
        engine=b.ASR.__new__(b.ASR)
        engine.spec=dict(model_id='synthetic/whisper');engine.revision='fixture'
        engine.device='cpu';engine.dtype='float32';engine.family='seq2seq';engine.max_new_tokens=400
        engine.sync=lambda:None;engine.processor=Processor();engine.model=Model()
        fake_torch=types.ModuleType('torch');fake_torch.inference_mode=contextlib.nullcontext
        fake_sf=types.ModuleType('soundfile');fake_sf.read=lambda *args,**kwargs:(np.zeros((16000,1),dtype='float32'),16000)
        with patch.dict('sys.modules',{'torch':fake_torch,'soundfile':fake_sf}),patch.object(b,'RSSMonitor',Monitor):
            result=engine.transcribe('synthetic.wav')
            self.assertEqual(result['status'],'ok')
            self.assertTrue(result['ended_with_eos'])
            engine.model.truncated=True
            result=engine.transcribe('synthetic.wav')
            self.assertEqual(result['status'],'error')
            self.assertIn('TOKEN_LIMIT_OR_MISSING_EOS',result['error_code'])
            self.assertFalse(result['ended_with_eos'])

    def test_qwen_adapter_waveform_prompt_slicing_and_output(self):
        import contextlib
        import types
        from unittest.mock import patch
        import numpy as np
        calls=[]
        class Tensor:
            def __init__(self,data): self.data=np.asarray(data)
            @property
            def shape(self): return self.data.shape
            def to(self,*args,**kwargs): return self
            def is_floating_point(self): return np.issubdtype(self.data.dtype,np.floating)
            def __getitem__(self,index): return Tensor(self.data[index])
            def tolist(self): return self.data.tolist()
        class Processor:
            def apply_transcription_request(self,**kwargs):
                calls.append(kwargs)
                return dict(input_ids=Tensor([[1,7,7,9]]),input_features=Tensor(np.zeros((1,8,16),dtype='float32')))
            def decode(self,ids,return_format):
                self.tokens=ids.tolist()
                return ['language Vietnamese<asr_text>synthetic transcript'] if return_format=='raw' else ['synthetic transcript']
        class Model:
            generation_config=types.SimpleNamespace(eos_token_id=[2,3],pad_token_id=2)
            truncated=False
            def generate(self,**kwargs):
                return types.SimpleNamespace(sequences=Tensor([[1,7,7,9,20,21] if self.truncated else [1,7,7,9,20,21,2]]))
        class Monitor:
            peak=1
            def __enter__(self): return self
            def __exit__(self,*args): pass
        engine=b.ASR.__new__(b.ASR)
        engine.spec=dict(model_id='synthetic/qwen');engine.revision='fixture'
        engine.device='cpu';engine.dtype='float32';engine.family='qwen3_asr';engine.max_new_tokens=512
        engine.sync=lambda:None;engine.processor=Processor();engine.model=Model()
        fake_torch=types.ModuleType('torch');fake_torch.inference_mode=contextlib.nullcontext
        fake_sf=types.ModuleType('soundfile');fake_sf.read=lambda *args,**kwargs:(np.zeros((16000,1),dtype='float32'),16000)
        with patch.dict('sys.modules',{'torch':fake_torch,'soundfile':fake_sf}),patch.object(b,'RSSMonitor',Monitor):
            result=engine.transcribe('synthetic.wav')
            self.assertEqual(result['status'],'ok')
            self.assertEqual(result['text_model'],'synthetic transcript')
            self.assertEqual(engine.processor.tokens,[[20,21,2]])
            self.assertIn('<asr_text>',result['text_output_raw'])
            self.assertEqual(set(calls[0]),{'audio','language'})
            self.assertEqual(calls[0]['language'],'vi')
            self.assertEqual(calls[0]['audio'].shape,(16000,))
            engine.model.truncated=True
            result=engine.transcribe('synthetic.wav')
            self.assertEqual(result['status'],'error')
            self.assertIn('TOKEN_LIMIT_OR_MISSING_EOS',result['error_code'])

    def test_checkpoint_includes_and_fingerprints_chat_template(self):
        import types
        from unittest.mock import patch
        calls=[]
        fake=types.ModuleType('huggingface_hub')
        fake.HfApi=lambda:types.SimpleNamespace(model_info=lambda *args,**kwargs:types.SimpleNamespace(sha='resolved_sha'))
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);checkpoint=out/'checkpoint';checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            template=checkpoint/'chat_template.jinja';template.write_text('first template')
            def snapshot(*args,**kwargs): calls.append(kwargs);return str(checkpoint)
            fake.snapshot_download=snapshot
            with patch.dict('sys.modules',{'huggingface_hub':fake}):
                spec=dict(name='qwen',model_id='synthetic/qwen',revision='main')
                b.resolve_checkpoint(spec,out)
                self.assertIn('*.jinja',calls[0]['allow_patterns'])
                spec['local_path']=str(checkpoint)
                _,first=b.resolve_checkpoint(spec,out,offline=True)
                template.write_text('changed template')
                _,second=b.resolve_checkpoint(spec,out,offline=True)
                self.assertNotEqual(first,second)

    def test_selection_table_constraints_and_pareto(self):
        base=dict(subset='all',complete=True,n=8,n_ok=8,n_missing=0,n_error=0,vram_peak_reserved_bytes=1024**3)
        summary=pd.DataFrame([dict(base,run='accurate',wer=.1,latency_p95_ms=200.),
            dict(base,run='fast',wer=.2,latency_p95_ms=100.),dict(base,run='dominated',wer=.3,latency_p95_ms=300.),
            dict(base,run='errors',wer=.01,latency_p95_ms=50.,n_ok=7,n_error=1)])
        with tempfile.TemporaryDirectory() as temp:
            result=b.selection_table(summary,{},Path(temp)/'selection.csv')
            flags=result.set_index('run')
            self.assertTrue(flags.loc['accurate','pareto_wer_latency'])
            self.assertTrue(flags.loc['fast','pareto_wer_latency'])
            self.assertFalse(flags.loc['dominated','pareto_wer_latency'])
            self.assertFalse(flags.loc['errors','eligible'])
            limited=b.selection_table(summary,{},Path(temp)/'limited.csv',max_latency_p95_ms=150.,max_vram_gb=2.)
            self.assertEqual(limited[limited.eligible].run.tolist(),['fast'])

    def test_full_denominator_and_no_success(self):
        rows=self.rows()
        records=[dict(ref_uid='a',status='ok',text_model='xin chào')]
        result=b.aggregate(b.score_rows(rows,records))
        self.assertEqual(result['wer'],.5)
        self.assertEqual(result['wer_success_only'],0.)
        self.assertEqual(result['failure_rate'],.5)
        self.assertFalse(result['complete'])
        empty=b.aggregate(b.score_rows(rows,[]))
        self.assertIsNone(empty['wer'])
        self.assertIsNone(empty['cer_no_spaces'])

    def test_journal_recovery_and_interior_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'predictions.jsonl'
            p.write_bytes(b'{"ref_uid":"a"}\n{"ref_')
            self.assertEqual(len(b.read_journal(p)),1)
            self.assertEqual(len(list(Path(temp).glob('*.torn-*'))),1)
            b.append_json(p,dict(ref_uid='b'))
            self.assertEqual(len(b.read_journal(p)),2)
            p.write_bytes(b'{bad}\n{"ref_uid":"a"}\n')
            with self.assertRaises(RuntimeError): b.read_journal(p)

    def test_review_preserved_on_extension_and_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            scored={'run':b.score_rows(self.rows(),[dict(ref_uid='a',status='ok',text_model='xin chào')])}
            b.make_review(scored,temp)
            p=Path(temp)/'critical_model_review_template.csv'
            d=pd.read_csv(p,keep_default_na=False)
            d.loc[d.ref_uid=='a','notes']='reviewed'
            d.to_csv(p,index=False)
            scored['run'].loc[scored['run'].ref_uid=='a','text_model']='changed after retry'
            b.make_review(scored,temp)
            kept=pd.read_csv(p,keep_default_na=False)
            self.assertEqual(len(kept),2)
            self.assertEqual(kept.loc[kept.ref_uid=='a','notes'].iloc[0],'reviewed')
            self.assertEqual(kept.loc[kept.ref_uid=='a','hypothesis'].iloc[0],'xin chào')

    def test_annotation_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'annotations.csv'
            b.save_template_preserving_rows(pd.DataFrame([dict(ref_uid='a',notes='done')]),p)
            b.save_template_preserving_rows(pd.DataFrame([dict(ref_uid='a',notes=''),dict(ref_uid='b',notes='')]),p)
            d=pd.read_csv(p,keep_default_na=False)
            self.assertEqual(len(d),2)
            self.assertEqual(d.iloc[0]['notes'],'done')

    def test_scoring_with_real_annotation_template_preserves_duration(self):
        rows=self.rows();rows['path']=['synthetic/a.wav','synthetic/b.wav']
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rd=out/'model_run';rd.mkdir()
            (rd/'config.json').write_text('{}')
            b.annotation_templates(rows,out)
            for row in rows.itertuples():
                b.append_json(rd/'predictions.jsonl',dict(ref_uid=row.ref_uid,status='ok',text_model=row.reference,latency_ms=row.duration*1000))
            blank,_=b.evaluate(rows,out)
            self.assertEqual(blank.loc[blank.subset=='all','wer'].iloc[0],0.)
            self.assertEqual(blank.loc[blank.subset=='accent_region=unannotated','rtf_weighted'].iloc[0],1.)
            labels=pd.read_csv(out/'dialect_annotations.csv',keep_default_na=False)
            labels.loc[0,'accent_region']='central'
            labels['duration']=99999.  # Review metadata must not replace validated scoring duration.
            labels['kind']='edited metadata'
            labels.to_csv(out/'dialect_annotations.csv',index=False)
            summary,scored=b.evaluate(rows,out)
            group=summary[summary.subset=='accent_region=central'].iloc[0]
            self.assertEqual(group['rtf_weighted'],1.)
            self.assertEqual(group['n_ok'],1)
            self.assertEqual(scored['model_run'].duration.tolist(),[1.,2.])


    def test_bootstrap_and_no_success_exclusion(self):
        records=[dict(ref_uid=x,status='ok',text_model=t) for x,t in [('a','xin chào'),('b','không gửi')]]
        d=b.score_rows(self.rows(),records)
        with tempfile.TemporaryDirectory() as temp:
            result=b.paired_bootstrap({'a':d,'b':d},Path(temp)/'paired.csv',n_boot=30)
            self.assertEqual(result[0]['ci95_low'],0.)
            self.assertEqual(result[0]['ci95_high'],0.)
            self.assertEqual(b.paired_bootstrap({'a':d,'b':b.score_rows(self.rows(),[])},Path(temp)/'empty.csv'),[])

    def test_paths_block_environment_and_relocate(self):
        with self.assertRaises(ValueError): b.safe_path('train/sample_environment/state.json')
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(b.resolve_audio('/old/train+public_test/train/a.wav',temp),Path(temp).resolve()/'train+public_test/train/a.wav')
            with self.assertRaises(ValueError): b.resolve_audio('/old/train+public_test/../../escape.wav',temp)

    def test_notebook_matches_module_and_compiles(self):
        root=Path(__file__).resolve().parents[1]
        module=(root/'asr_benchmark.py').read_text(encoding='utf-8')
        n,cells=self.notebook_cells()
        literal=ast.parse(cells['implementation']).body[0].value
        self.assertEqual(ast.literal_eval(literal),module)
        self.assertIn(hashlib.sha256(module.encode()).hexdigest(),cells['implementation'])
        for i,c in enumerate(n['cells']):
            if c['cell_type']=='code': compile(''.join(c['source']),f'cell_{i}','exec')

    def test_export_allowlist_and_partial_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'output';root.mkdir()
            (root/'metrics.csv').write_text('n,n_ok\n2,1\n')
            (root/'original.wav').write_bytes(b'not real audio')
            (root/'weights.safetensors').write_bytes(b'not real weights')
            blocked=root/'sample_environment';blocked.mkdir()
            (blocked/'state.json').write_text('{}')
            run=root/'model_run';run.mkdir()
            (run/'config.json').write_text('{}')
            (run/'predictions.jsonl').write_text('{"status":"error"}\n')
            (run/'audio.wav').write_bytes(b'not real audio')
            path=b.package_outputs(root,Path(temp)/'results')
            with zipfile.ZipFile(path) as z:
                self.assertEqual(set(z.namelist()),{'metrics.csv','model_run/config.json','model_run/predictions.jsonl','EXPORT_MANIFEST.json'})
                manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
                self.assertIn('comparison_status.csv',manifest['missing_summary_files'])
                for entry in manifest['files']:
                    self.assertEqual(hashlib.sha256(z.read(entry['path'])).hexdigest(),entry['sha256'])

    def test_notebook_status_includes_failed_checkpoint(self):
        _,cells=self.notebook_cells()
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rd=out/'model_a';rd.mkdir()
            (rd/'config.json').write_text(json.dumps(dict(model_id='model/a')))
            for uid,text in [('a','xin chào'),('b','không gửi')]:
                b.append_json(rd/'predictions.jsonl',dict(ref_uid=uid,status='ok',text_model=text))
            env=dict(vars(b),rows=self.rows(),OUTPUT_DIR=out,STAGE='benchmark',AUDIO_ROOT=out,
                     DEVICE='cpu',PRECISION='float32',OFFLINE=True,RETRY_ERRORS=False,MAX_NEW_TOKENS=400,
                     MAX_LATENCY_P95_MS=None,MAX_VRAM_GB=None,
                     BENCHMARK_SOURCE='synthetic source',BENCHMARK_SOURCE_SHA256='synthetic hash',cache_hashes={},
                     CANDIDATES=[dict(name='a',model_id='model/a'),dict(name='b',model_id='model/b'),dict(name='c',model_id='model/c')],
                     SELECTED_MODELS=['a','b','c'],
                     selected=[dict(name='a',model_id='model/a'),dict(name='b',model_id='model/b'),dict(name='c',model_id='model/c')],
                     display=lambda x:None,print=lambda *args,**kwargs:None)
            exec(cells['plan'],env)
            env['run_dirs']=[rd]
            exec(cells['comparison'],env)
            status=pd.read_csv(out/'comparison_status.csv')
            self.assertEqual(len(status),3)
            self.assertEqual(status.iloc[0]['state'],'ready')
            self.assertEqual(status.iloc[1]['state'],'not_run_or_checkpoint_error')
            self.assertEqual(status.iloc[2]['n_missing'],2)

    def test_standalone_zip_cell_and_selection_defaults(self):
        import types
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        _,cells=self.notebook_cells()
        config={};exec(cells['config'],config)
        self.assertEqual(config['STAGE'],'smoke')
        self.assertEqual(config['SELECTED_MODELS'],['phowhisper_medium','whisper_turbo','qwen3_asr_06b','qwen3_asr_17b','wav2vec2_vi'])
        with tempfile.TemporaryDirectory(dir=root) as temp:
            out=Path(temp)/'output';out.mkdir()
            (out/'model_errors.jsonl').write_text('{"error":"synthetic"}\n')
            display=types.ModuleType('IPython.display')
            display.FileLink=lambda x:x;display.display=lambda x:None
            with patch.dict('sys.modules',{'IPython.display':display}):
                env=dict(OUTPUT_DIR=out,print=lambda *args,**kwargs:None)
                exec(cells['export'],env)
            with zipfile.ZipFile(env['zip_path']) as z:
                self.assertIn('model_errors.jsonl',z.namelist())

    def test_followup_nested_root_and_zip(self):
        names=['references_enriched.csv','split_manifest.csv','benchmark_manifest.csv']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'followup';nested=root/'lifemate_asr_followup';nested.mkdir(parents=True)
            for name in names: (nested/name).write_text('ref_uid\nsynthetic\n')
            cache,meta=b.resolve_followup_cache(root,Path(temp)/'work')
            self.assertEqual(cache,nested.resolve())
            for name in names: (nested/name).unlink()
            archive=root/'lifemate_asr_followup.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name in names:
                    z.writestr('lifemate_asr_followup/'+name,'ref_uid\nsynthetic\n')
                    z.writestr('sample_environment/'+name,'forbidden fixture')
                z.writestr('sample_environment/state.json','forbidden fixture')
            cache,meta=b.resolve_followup_cache(root,Path(temp)/'work')
            self.assertEqual(meta['mode'],'three_csvs_from_zip')
            self.assertEqual({p.name for p in cache.iterdir()},set(names))
            archive.unlink()
            with self.assertRaises(FileNotFoundError): b.resolve_followup_cache(root,Path(temp)/'work')

    def test_selected_audio_extraction_and_mounted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'dataset';root.mkdir()
            rows=pd.DataFrame([dict(path='/old/train+public_test/train/audio/a.wav')])
            with zipfile.ZipFile(root/'train+public_test.zip','w') as z:
                z.writestr('train+public_test/train/audio/a.wav',b'synthetic wav placeholder')
                z.writestr('train+public_test/train/audio/b.wav',b'not selected')
                z.writestr('train+public_test/sample_environment/state.json',b'forbidden fixture')
            dest,meta=b.prepare_selected_audio(rows,root,Path(temp)/'work')
            self.assertEqual(meta['mode'],'selected_zip_entries')
            self.assertEqual([p.name for p in dest.rglob('*') if p.is_file()],['a.wav'])
            dest2,meta2=b.prepare_selected_audio(rows,dest,Path(temp)/'work')
            self.assertEqual(meta2['mode'],'mounted_files')
            self.assertEqual(dest,dest2)
            rows.loc[0,'path']='/old/train+public_test/train/audio/missing.wav'
            with self.assertRaises(FileNotFoundError): b.prepare_selected_audio(rows,root,Path(temp)/'work')

    def test_separate_model_cells_execute_same_rows_without_real_models(self):
        _,cells=self.notebook_cells()
        config={};exec(cells['config'],config)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);calls=[]
            def fake_runner(rows,specs,cfg,hashes):
                calls.append((specs[0]['model_id'],tuple(rows.ref_uid)))
                d=out/specs[0]['name'];d.mkdir(exist_ok=True)
                (d/'config.json').write_text(json.dumps(dict(model_id=specs[0]['model_id'])))
                for row in rows.itertuples():
                    b.append_json(d/'predictions.jsonl',dict(ref_uid=row.ref_uid,status='ok',text_model=row.reference))
                return [d]
            env=dict(vars(b),**{k:v for k,v in config.items() if k!='__builtins__'})
            env.update(rows=self.rows(),OUTPUT_DIR=out,AUDIO_ROOT=out,cache_hashes={},
                BENCHMARK_SOURCE='synthetic source',BENCHMARK_SOURCE_SHA256='synthetic hash',
                run_benchmark=fake_runner,display=lambda x:None,print=lambda *args,**kwargs:None)
            env['rows']['path']=['synthetic/a.wav','synthetic/b.wav']
            b.annotation_templates(env['rows'],out)
            exec(cells['plan'],env)
            for role in [name for name in cells if name.startswith('model_')]: exec(cells[role],env)
            exec(cells['comparison'],env)
            self.assertEqual([x[0] for x in calls],[s['model_id'] for s in config['CANDIDATES'] if s['name'] in config['SELECTED_MODELS']])
            self.assertTrue(all(x[1]==('a','b') for x in calls))
            self.assertTrue(env['comparison_status'].state.eq('ready').all())
            self.assertEqual(len(env['summary'][env['summary'].subset=='all']),5)
            self.assertEqual(len(pd.read_csv(out/'paired_bootstrap.csv')),10)
            table=pd.read_csv(out/'model_selection.csv')
            self.assertEqual(len(table),5)
            self.assertTrue(table.human_review_required.all())

    def test_input_cell_with_full_synthetic_cache_and_zip(self):
        import types
        from unittest.mock import patch
        _,cells=self.notebook_cells()
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);followup=base/'followup';cache=followup/'lifemate_asr_followup'
            cache.mkdir(parents=True)
            refs=pd.DataFrame([dict(ref_uid=f'r{i:05d}',sample_uid=f's{i:05d}',
                path=f'/old/train+public_test/{"train" if i<5036 else "public_test"}/audio/{i}.wav',
                split='train' if i<5036 else 'test',
                manifest='/old/train+public_test/train/train.jsonl' if i<5036 else '/old/train+public_test/public_test/public_test.jsonl',
                kind='final' if i%2 else 'history',duration=1.,reference='synthetic reference' if i<5036 else '') for i in range(6969)])
            split=refs.iloc[:5036].copy()
            split['partition']=['development']*4045+['validation']*991
            split['group_id']=split.sample_uid
            benchmark=split.iloc[4045:4145].copy()
            for d,name in [(refs,'references_enriched.csv'),(split,'split_manifest.csv'),(benchmark,'benchmark_manifest.csv')]: d.to_csv(cache/name,index=False)
            dataset=base/'dataset';dataset.mkdir()
            with zipfile.ZipFile(dataset/'data.zip','w') as z:
                for row in benchmark.itertuples():
                    z.writestr(row.path.split('/old/')[1],b'synthetic audio bytes, not real WAV')
            display=types.ModuleType('IPython.display');display.display=lambda x:None
            env=dict(vars(b),CACHE_DIR=None,FOLLOWUP_INPUT=followup,WORK_DIR=base/'work',STAGE='benchmark',
                OUTPUT_DIR=base/'output',RESUME_DIR=None,RUN_INFERENCE=True,DATASET_ROOT=dataset,
                DATASET_ZIP=None,AUDIT_INPUT=base/'audit',print=lambda *args,**kwargs:None)
            with patch.dict('sys.modules',{'IPython.display':display}): exec(cells['inputs'],env)
            self.assertEqual(len(env['rows']),100)
            self.assertEqual(env['CACHE_DIR'],cache.resolve())
            self.assertEqual(env['audio_provenance']['mode'],'selected_zip_entries')
            self.assertEqual(len(list(env['AUDIO_ROOT'].rglob('*.wav'))),100)
            self.assertTrue((base/'output'/'input_provenance.json').is_file())


if __name__=='__main__': unittest.main()
