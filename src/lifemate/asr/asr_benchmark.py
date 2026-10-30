"""LifeMate cached ASR benchmark. No dataset crawl, labels or external audio API."""
from pathlib import Path
import json, hashlib, os, re, unicodedata, time, platform, threading, traceback, gc, shutil
import numpy as np
import pandas as pd

NORMALIZATION = 'nfc_lower_punct_symbol_to_space_keep_accents_digits_v1'
CODE_VERSION = 'lifemate-model-selection-2.0'

def normalize(text):
    text = unicodedata.normalize('NFC', str(text)).lower()
    text = ''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in text)
    return ' '.join(text.split())

def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def safe_path(path):
    p = Path(path)
    if any(part.endswith('_environment') for part in p.parts):
        raise ValueError('Environment access forbidden: '+str(p))
    return p

def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));tmp.replace(path)

def append_json(path,obj):
    with open(path,'a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,default=str,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def read_journal(path):
    """Recover only a torn final line; malformed interior records stop the run."""
    path=Path(path)
    if not path.exists(): return []
    raw=path.read_bytes();lines=raw.splitlines(keepends=True);rows=[];offset=0
    for i,line in enumerate(lines):
        try: row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if i!=len(lines)-1: raise RuntimeError('Corrupt journal interior: '+str(path))
            backup=path.with_name(path.name+'.torn-'+str(time.time_ns()))
            backup.write_bytes(raw)
            path.write_bytes(raw[:offset])
            append_json(path.parent/'recovery.jsonl',{'journal':path.name,'backup':backup.name,'discarded_bytes':len(raw)-offset})
            break
        rows.append(row);offset+=len(line)
    # A valid final JSON without newline must be terminated before appending.
    if rows and path.read_bytes() and not path.read_bytes().endswith(b'\n'):
        with open(path,'ab') as f: f.write(b'\n');f.flush();os.fsync(f.fileno())
    return rows

def resolve_followup_cache(input_root,working_dir,override=None):
    """Read the named follow-up mount only; never fall back to old audit metadata."""
    import zipfile
    root=safe_path(Path(input_root).resolve())
    names=['references_enriched.csv','split_manifest.csv','benchmark_manifest.csv']
    candidates=[safe_path(Path(override).resolve())] if override is not None else [root/'lifemate_asr_followup',root]
    for p in candidates:
        safe_path(p)
        if all((p/n).is_file() for n in names):
            return p,dict(source=str(p),mode='csv_directory')
    if override is not None:
        raise FileNotFoundError('CACHE_DIR must directly contain the three CSVs: '+str(override))
    archive=root/'lifemate_asr_followup.zip'
    if not archive.is_file():
        raise FileNotFoundError('Follow-up cache missing. Add the saved notebook output at '+str(root)+
            '; expected lifemate_asr_followup/{'+','.join(names)+'} or lifemate_asr_followup.zip.')
    with zipfile.ZipFile(archive) as z:
        members={}
        for info in z.infolist():
            parts=info.filename.replace('\\','/').split('/')
            if info.is_dir() or '__MACOSX' in parts or any(x.endswith('_environment') for x in parts): continue
            if parts[-1] in names:
                if any(x in ['..',''] for x in parts) or info.filename.startswith('/'):
                    raise ValueError('Unsafe cache ZIP member')
                if parts[-1] in members: raise ValueError('Ambiguous cache CSV in ZIP: '+parts[-1])
                members[parts[-1]]=info
        if set(members)!=set(names): raise FileNotFoundError('Follow-up ZIP must contain one of each required CSV')
        dest=safe_path(Path(working_dir).resolve()/'followup_cache');dest.mkdir(parents=True,exist_ok=True)
        for name,info in members.items():
            target=dest/name
            if target.is_symlink(): raise ValueError('Symlink cache target forbidden')
            target.write_bytes(z.read(info))
    return dest,dict(source=str(archive),mode='three_csvs_from_zip')


def prepare_selected_audio(rows,dataset_root,working_dir,archive_path=None):
    """Use selected files or extract only their exact entries from the dataset ZIP."""
    import zipfile
    root=safe_path(Path(dataset_root).resolve())
    if all(resolve_audio(x,root).is_file() for x in rows.path):
        return root,dict(source=str(root),mode='mounted_files',n_selected=len(rows))
    archives=[safe_path(Path(archive_path).resolve())] if archive_path is not None else [
        p for p in [root/'train+public_test.zip',root/'data.zip'] if p.is_file()]
    if len(archives)!=1:
        raise FileNotFoundError('Selected audio missing under '+str(root)+
            '. Mount extracted train+public_test/ or set DATASET_ZIP to the exact ZIP path.')
    archive=archives[0]
    dest=safe_path(Path(working_dir).resolve()/'selected_audio');dest.mkdir(parents=True,exist_ok=True)
    requested={resolve_audio(x,dest).relative_to(dest).as_posix() for x in rows.path}
    with zipfile.ZipFile(archive) as z:
        members={}
        for info in z.infolist():
            parts=info.filename.replace('\\','/').split('/')
            if info.is_dir() or '__MACOSX' in parts or any(x.endswith('_environment') for x in parts): continue
            if info.filename not in requested: continue
            if info.filename in members: raise ValueError('Duplicate selected audio entry in ZIP')
            members[info.filename]=info
        if set(members)!=requested:
            raise FileNotFoundError(f'ZIP missing {len(requested-set(members))} selected audio entries; expected train+public_test/... paths')
        required=sum(i.file_size for i in members.values())
        if shutil.disk_usage(dest).free < required+16*1024*1024: raise OSError('Not enough space for selected audio')
        for name,info in members.items():
            target=resolve_audio(name,dest)
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_suffix(target.suffix+'.tmp')
            if tmp.is_symlink(): raise ValueError('Symlink extraction target forbidden')
            try:
                with z.open(info) as src,open(tmp,'wb') as out: shutil.copyfileobj(src,out)
                tmp.replace(target)
            finally:
                if tmp.exists(): tmp.unlink()
    return dest,dict(source=str(archive),mode='selected_zip_entries',n_selected=len(rows),bytes_selected=required)


def load_cache(cache_dir):
    p=safe_path(cache_dir)
    names=['references_enriched.csv','split_manifest.csv','benchmark_manifest.csv']
    for n in names:
        if not (p/n).is_file(): raise FileNotFoundError(f'Missing cache {p/n}; set CACHE_DIR to the directory containing the three CSV files.')
    r,s,b=[pd.read_csv(p/n,keep_default_na=False) for n in names]
    for d in [r,s,b]:
        if d.ref_uid.duplicated().any(): raise ValueError('Duplicate ref_uid')
    if not set(s.partition)<= {'development','validation'}: raise ValueError('Unexpected partition')
    j=s[['ref_uid','partition','group_id']].merge(r,on='ref_uid',validate='one_to_one',how='left',indicator=True)
    if not j._merge.eq('both').all(): raise ValueError('Split IDs missing from reference cache')
    if not j.split.eq('train').all() or not j.manifest.map(lambda x:Path(x).name=='train.jsonl').all():
        raise ValueError('Public test or non-train manifest in labeled evaluation')
    # Compare by ID, never infer pairing by row order.
    by=r.set_index('ref_uid')
    for d in [s,b]:
        if not set(d.ref_uid)<=set(by.index): raise ValueError('Unmapped ID')
        for col in ['path','reference','sample_uid','kind']:
            if col in d and not (d[col].to_numpy()==by.loc[d.ref_uid,col].to_numpy()).all():
                raise ValueError('Cache disagreement: '+col)
    dev=j[j.partition=='development'];val=j[j.partition=='validation']
    if set(dev.sample_uid)&set(val.sample_uid) or set(dev.group_id)&set(val.group_id): raise ValueError('Group leakage')
    if not set(b.ref_uid)<=set(val.ref_uid): raise ValueError('Benchmark not validation-only')
    if len(r)!=6969 or len(s)!=5036 or len(val)!=991 or len(b)!=100: raise ValueError('Unexpected cache version; inspect explicitly before changing expected counts')
    if val.reference.map(lambda x:not isinstance(x,str) or not x.strip()).any(): raise ValueError('Missing gold text')
    for x in r.path: safe_path(x)
    hashes={n:digest_file(p/n) for n in names}
    return r,j.drop(columns='_merge'),b,hashes

def select_rows(joined,benchmark,stage):
    if stage=='validation': rows=joined[joined.partition=='validation'].copy().sort_values('ref_uid')
    elif stage in ['smoke','benchmark']:
        rows=joined.set_index('ref_uid').loc[benchmark.ref_uid].reset_index()
        if stage=='smoke':
            # Deterministic endpoints per kind exercise short/long final/history; do not reshuffle benchmark.
            picks=[]
            for kind in ['final','history']:
                sub=rows[rows.kind==kind].sort_values(['duration','ref_uid'])
                picks+=sub.head(2).ref_uid.tolist()+sub.tail(2).ref_uid.tolist()
            rows=rows[rows.ref_uid.isin(picks)].copy()
    else: raise ValueError('stage must be smoke / benchmark / validation')
    return rows.reset_index(drop=True)

def resolve_audio(old_path,dataset_root):
    old=safe_path(old_path)
    parts=old.parts
    if 'train+public_test' not in parts: raise ValueError('No known root marker: '+str(old))
    tail=Path(*parts[parts.index('train+public_test'):])
    root=safe_path(dataset_root).resolve()
    target=safe_path(root/tail).resolve()
    if not target.is_relative_to(root): raise ValueError('Audio escaped configured root')
    safe_path(target)
    return target

def runtime_info():
    import torch,transformers,soundfile,psutil,importlib.metadata as md
    versions={n:md.version(n) for n in ['torch','transformers','numpy','pandas','soundfile','psutil','huggingface-hub','tokenizers','safetensors']}
    return dict(versions=versions,python=platform.python_version(),platform=platform.platform(),cpu=platform.processor(),
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                cpu_threads=torch.get_num_threads())

def resolve_checkpoint(spec,out,offline=False):
    """Pin remote branch to SHA once; local mounted checkpoint gets content fingerprint."""
    from huggingface_hub import HfApi,snapshot_download
    lock=out/(spec['name']+'_checkpoint.json')
    old=json.loads(lock.read_text()) if lock.exists() else None
    if spec.get('local_path'):
        source=safe_path(spec['local_path']).resolve()
        if not source.is_dir(): raise FileNotFoundError('Checkpoint mount missing: '+str(source))
        if not (source/'config.json').is_file(): raise ValueError('Local checkpoint must contain config.json; no scan performed')
        entries=[]
        # Only checkpoint files, never dataset audio or environment directories.
        for root,dirs,files in os.walk(source):
            dirs[:]=sorted(d for d in dirs if not d.endswith('_environment') and d not in ['.git','.cache'])
            for name in sorted(files):
                f=safe_path((Path(root)/name).resolve())
                if f.suffix in ['.json','.bin','.safetensors','.txt','.model','.jinja']:
                    entries.append((str(f.relative_to(source)),digest_file(f)))
        if not entries or not (source/'config.json').exists(): raise ValueError('Incomplete local checkpoint')
        identity='local-sha256:'+fingerprint(entries)
        obj=dict(model_id=spec['model_id'],revision=identity,requested_revision=spec.get('revision','main'),local_files=entries)
    else:
        if offline and not old: raise ValueError('OFFLINE requires local_path or a previously saved checkpoint lock and HF cache')
        sha=old['revision'] if old and old.get('model_id')==spec['model_id'] and old.get('requested_revision')==spec.get('revision','main') else None
        if offline and sha is None: raise ValueError('Checkpoint lock does not match offline request')
        if sha is None: sha=HfApi().model_info(spec['model_id'],revision=spec.get('revision','main')).sha
        source=Path(snapshot_download(spec['model_id'],revision=sha,local_files_only=offline,
                allow_patterns=['*.json','*.txt','*.model','*.jinja','*.safetensors','pytorch_model*.bin']))
        identity=sha;obj=dict(model_id=spec['model_id'],revision=sha,requested_revision=spec.get('revision','main'))
    write_json(lock,obj)
    return source,identity

class RSSMonitor:
    """Sample process RSS every 10 ms; brief peaks may be missed; not system RAM."""
    def __enter__(self):
        import psutil
        self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.01): self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()

def whisper_sequence_diagnostics(sequence,eos_token_id,pad_token_id=None):
    """Check raw generate sequences, not Whisper's EOS-stripped segment tokens."""
    tokens=sequence.tolist() if hasattr(sequence,'tolist') else list(sequence)
    eos=set(eos_token_id if isinstance(eos_token_id,(list,tuple,set)) else [eos_token_id])
    if None in eos or not eos: raise ValueError('Whisper EOS token is not configured')
    # Some generation configurations use a separate pad token after EOS.
    while tokens and tokens[-1]==pad_token_id and pad_token_id not in eos: tokens.pop()
    return dict(output_token_count=len(tokens),ended_with_eos=bool(tokens and tokens[-1] in eos))


class ASR:
    def __init__(self,spec,source,revision,device='cuda:0',precision='float16',max_new_tokens=400):
        import torch
        self.spec=spec;self.revision=revision;self.device=device;self.family=spec['family'];self.max_new_tokens=max_new_tokens
        self.dtype=torch.float16 if precision=='float16' else torch.float32
        if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('Requested CUDA unavailable; select Kaggle GPU or explicitly set DEVICE=cpu and PRECISION=float32')
        if device=='cpu' and precision!='float32': raise ValueError('Use float32 on CPU')
        if self.family=='seq2seq':
            from transformers import WhisperProcessor as proc,WhisperForConditionalGeneration as model
        elif self.family=='ctc':
            from transformers import Wav2Vec2Processor as proc,Wav2Vec2ForCTC as model
        elif self.family=='qwen3_asr':
            from transformers import AutoProcessor as proc,Qwen3ASRForConditionalGeneration as model
        else: raise ValueError('Unsupported model family: '+str(self.family))
        self.processor=proc.from_pretrained(str(source),local_files_only=True)
        self.model=model.from_pretrained(str(source),local_files_only=True,dtype=self.dtype,attn_implementation='eager').to(device).eval()
        if self.family=='seq2seq': self.model.generation_config.forced_decoder_ids=None
    def sync(self):
        import torch
        if self.device.startswith('cuda'): torch.cuda.synchronize(self.device)
    def transcribe(self,audio_path):
        """Only audio enters inference. No reference, context, tool schema or confidence."""
        import torch,soundfile as sf
        result=dict(status='error',text_model='',text_normalized='',model_id=self.spec['model_id'],revision=self.revision,
                    latency_ms=None,model_ms=None,duration_s=None,rtf=None,warnings=[],error_code=None)
        self.sync();start=time.perf_counter();mon=None
        try:
            if self.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(self.device)
            with RSSMonitor() as mon:
                wav,sr=sf.read(safe_path(audio_path),dtype='float32',always_2d=True)
                if sr!=16000 or wav.shape[1]!=1: raise ValueError('UNEXPECTED_AUDIO_FORMAT: expected mono 16000Hz')
                wav=wav[:,0];duration=len(wav)/sr;result['duration_s']=duration
                if not len(wav) or not np.isfinite(wav).all(): raise ValueError('INVALID_AUDIO')
                if self.family=='seq2seq' and duration>30: raise ValueError('LONG_AUDIO_UNSUPPORTED: no truncation permitted')
                if self.family=='ctc' and duration>10: result['warnings'].append('above_model_card_10s_recommendation_full_audio_used')
                if self.family=='seq2seq':
                    batch=self.processor(wav,sampling_rate=sr,return_tensors='pt',return_attention_mask=True,truncation=False)
                    inputs={k:v.to(self.device,dtype=self.dtype) if k=='input_features' else v.to(self.device) for k,v in batch.items()}
                elif self.family=='ctc':
                    batch=self.processor(wav,sampling_rate=sr,return_tensors='pt',padding=False)
                    inputs={k:v.to(self.device,dtype=self.dtype) if k=='input_values' else v.to(self.device) for k,v in batch.items()}
                elif self.family=='qwen3_asr':
                    # Native HF processor accepts the waveform already validated above.
                    # Language is fixed globally; no transcript/context/hotwords are passed.
                    # Pinned native processor defaults to 16 kHz and PyTorch tensors.
                    batch=self.processor.apply_transcription_request(audio=wav,language='vi')
                    inputs={k:v.to(device=self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k,v in batch.items()}
                self.sync();model_start=time.perf_counter()
                with torch.inference_mode():
                    if self.family=='seq2seq':
                        # HF Whisper's default tensor output strips EOS during segment processing.
                        # The structured short-form output retains raw sequences including EOS.
                        generated=self.model.generate(**inputs,language='vi',task='transcribe',num_beams=1,do_sample=False,
                                                max_new_tokens=self.max_new_tokens,return_timestamps=False,
                                                return_dict_in_generate=True,output_scores=False,output_logits=False)
                        ids=generated.sequences
                    elif self.family=='ctc': ids=self.model(**inputs).logits.argmax(dim=-1)
                    else:
                        generated=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,num_beams=1,do_sample=False,
                            return_dict_in_generate=True,output_scores=False,output_logits=False)
                        ids=generated.sequences[:,inputs['input_ids'].shape[1]:]
                self.sync();result['model_ms']=(time.perf_counter()-model_start)*1000
                if self.family=='qwen3_asr':
                    raw=self.processor.decode(ids,return_format='raw')
                    parsed=self.processor.decode(ids,return_format='transcription_only')
                    result['text_output_raw']=raw[0] if isinstance(raw,list) else raw
                    text=parsed[0] if isinstance(parsed,list) else parsed
                    if '<asr_text>' not in result['text_output_raw']: raise RuntimeError('QWEN_OUTPUT_SCHEMA_MISSING_ASR_TEXT')
                else: text=self.processor.batch_decode(ids,skip_special_tokens=True)[0]
                result['text_model']=text;result['text_normalized']=normalize(text)
                if self.family in ['seq2seq','qwen3_asr']:
                    diagnostics=whisper_sequence_diagnostics(ids[0],self.model.generation_config.eos_token_id,
                        self.model.generation_config.pad_token_id)
                    result.update(diagnostics)
                    # Keep partial text for diagnosis, but never score a genuinely unfinished sequence as success.
                    if not diagnostics['ended_with_eos']: raise RuntimeError('TOKEN_LIMIT_OR_MISSING_EOS')
                if not text.strip(): raise RuntimeError('EMPTY_TRANSCRIPT')
                result['status']='ok'
        except Exception as exc:
            result['error_code']=type(exc).__name__+': '+str(exc)
        finally:
            self.sync();result['latency_ms']=(time.perf_counter()-start)*1000
            if result['duration_s']: result['rtf']=result['latency_ms']/1000/result['duration_s']
            result['rss_peak_bytes']=mon.peak if mon else None
            result['vram_peak_allocated_bytes']=torch.cuda.max_memory_allocated(self.device) if self.device.startswith('cuda') else None
            result['vram_peak_reserved_bytes']=torch.cuda.max_memory_reserved(self.device) if self.device.startswith('cuda') else None
        return result

def run_benchmark(rows,specs,cfg,cache_hashes):
    import torch
    out=safe_path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    env=runtime_info();write_json(out/'runtime.json',env)
    torch.manual_seed(2026);np.random.seed(2026)
    # preflight only selected audio paths, no decoding/hash/scan; fail clearly on missing mounts.
    if not safe_path(cfg['dataset_root']).is_dir(): raise FileNotFoundError('DATASET_ROOT missing: '+str(cfg['dataset_root']))
    paths={x.ref_uid:str(resolve_audio(x.path,cfg['dataset_root'])) for x in rows.itertuples()}
    print('Selected audio paths missing:',sum(not Path(p).is_file() for p in paths.values()),'(each is recorded as a sample error)')
    write_json(out/'selection.json',dict(stage=cfg['stage'],ref_uids=rows.ref_uid.tolist(),cache_hashes=cache_hashes))
    run_dirs=[]
    for spec in specs:
        engine=None
        try:
            source,revision=resolve_checkpoint(spec,out,cfg.get('offline',False))
            token_limit=spec.get('max_new_tokens',cfg['max_new_tokens'])
            identity=dict(code=CODE_VERSION,source_sha256=cfg['source_sha256'],model_id=spec['model_id'],revision=revision,
                          family=spec['family'],device=cfg['device'],precision=cfg['precision'],max_new_tokens=token_limit,
                          decoding='greedy_vi_no_reference_no_context_no_external_LM',normalization=NORMALIZATION,cache_hashes=cache_hashes,
                          runtime=env)
            run_id=fingerprint(identity)[:20];rd=out/(spec['name']+'_'+run_id);rd.mkdir(exist_ok=True);run_dirs.append(rd)
            write_json(rd/'config.json',identity);journal=rd/'predictions.jsonl'
            prev=read_journal(journal)
            latest={x['ref_uid']:x for x in prev}
            completed={uid for uid,x in latest.items() if x['status']=='ok' or not cfg.get('retry_errors',False)}
            pending=rows[~rows.ref_uid.isin(completed)]
            if pending.empty: print(spec['name'],'already completed');continue
            with RSSMonitor() as load_mon:
                start=time.perf_counter();engine=ASR(spec,source,revision,cfg['device'],cfg['precision'],token_limit);engine.sync()
                load_ms=(time.perf_counter()-start)*1000
            write_json(rd/'model_config.json',engine.model.config.to_dict())
            if getattr(engine.model,'generation_config',None) is not None:
                write_json(rd/'generation_config.json',engine.model.generation_config.to_dict())
            session_id=str(time.time_ns())
            write_json(rd/('session_'+session_id+'.json'),dict(session_id=session_id,load_ms=load_ms,load_rss_peak_bytes=load_mon.peak,hardware=env))
            warm=engine.transcribe(paths[pending.iloc[0].ref_uid]);write_json(rd/('warmup_'+session_id+'.json'),warm)
            if warm['status']!='ok': print('Warmup failed (retained):',warm['error_code'])
            for i,row in enumerate(pending.itertuples()):
                record=engine.transcribe(paths[row.ref_uid])
                if record.get('duration_s') is not None and abs(record['duration_s']-float(row.duration)) > 1/16000:
                    record['status']='error';record['error_code']='CACHE_AUDIO_DURATION_MISMATCH'
                record.update(ref_uid=row.ref_uid,session_id=session_id,run_id=run_id)
                append_json(journal,record)
                if i%10==0: print(spec['name'],i+1,'/',len(pending),record['status'])
                if record['status']!='ok' and 'out of memory' in str(record['error_code']).lower(): torch.cuda.empty_cache()
        except Exception:
            err=dict(model=spec['name'],stage=cfg['stage'],traceback=traceback.format_exc(),time_ns=time.time_ns())
            append_json(out/'model_errors.jsonl',err);print('MODEL ERROR:',spec['name'],err['traceback'])
        finally:
            del engine;gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return run_dirs

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def score_rows(rows,records):
    latest={x['ref_uid']:x for x in records};out=[]
    for row in rows.itertuples():
        p=latest.get(row.ref_uid);status=p['status'] if p else 'missing'
        ref=normalize(row.reference);hyp=normalize(p['text_model']) if p and status=='ok' else ''
        # Failure-as-empty keeps every selected reference in the denominator; also report success-only.
        rc=ref.replace(' ','');hc=hyp.replace(' ','')
        out.append(dict(ref_uid=row.ref_uid,sample_uid=row.sample_uid,kind=row.kind,duration=float(row.duration),status=status,
                        reference=row.reference,text_model=p.get('text_model','') if p else '',word_edits=edit_distance(ref.split(),hyp.split()),
                        n_words=len(ref.split()),char_edits=edit_distance(rc,hc),n_chars=len(rc),
                        latency_ms=p.get('latency_ms') if p else None,rtf=p.get('rtf') if p else None,
                        rss_peak_bytes=p.get('rss_peak_bytes') if p else None,
                        vram_peak_allocated_bytes=p.get('vram_peak_allocated_bytes') if p else None,
                        vram_peak_reserved_bytes=p.get('vram_peak_reserved_bytes') if p else None,
                        warnings=json.dumps(p.get('warnings',[])) if p else '[]'))
    return pd.DataFrame(out)

def aggregate(d):
    ok=d[d.status=='ok'];timed=ok.dropna(subset=['latency_ms'])
    return dict(n=len(d),n_ok=len(ok),n_error=int(d.status.eq('error').sum()),n_missing=int(d.status.eq('missing').sum()),
        complete=bool(not d.status.eq('missing').any()),failure_rate=float(d.status.ne('ok').mean()),
        wer=float(d.word_edits.sum()/d.n_words.sum()) if len(ok) and d.n_words.sum() else None,
        cer_no_spaces=float(d.char_edits.sum()/d.n_chars.sum()) if len(ok) and d.n_chars.sum() else None,
        wer_success_only=float(ok.word_edits.sum()/ok.n_words.sum()) if ok.n_words.sum() else None,
        cer_success_only=float(ok.char_edits.sum()/ok.n_chars.sum()) if ok.n_chars.sum() else None,
        latency_p50_ms=float(timed.latency_ms.median()) if len(timed) else None,
        latency_p95_ms=float(timed.latency_ms.quantile(.95)) if len(timed) else None,
        rtf_weighted=float((timed.latency_ms.sum()/1000)/timed.duration.sum()) if len(timed) else None,
        rss_peak_bytes=int(ok.rss_peak_bytes.max()) if ok.rss_peak_bytes.notna().any() else None,
        vram_peak_allocated_bytes=int(ok.vram_peak_allocated_bytes.max()) if ok.vram_peak_allocated_bytes.notna().any() else None,
        vram_peak_reserved_bytes=int(ok.vram_peak_reserved_bytes.max()) if ok.vram_peak_reserved_bytes.notna().any() else None)

def evaluate(rows,out_dir,run_dirs=None):
    out=Path(out_dir);summaries=[];scored={}
    # Enumerate only immediate benchmark output children, never dataset.
    for rd in sorted(out.iterdir() if run_dirs is None else set(map(Path,run_dirs))):
        if not rd.is_dir() or not (rd/'config.json').is_file(): continue
        d=score_rows(rows,read_journal(rd/'predictions.jsonl'));d.to_csv(rd/'per_audio.csv',index=False);scored[rd.name]=d
        groups={'all':d,'final':d[d.kind=='final'],'history':d[d.kind=='history'],
                'duration_le10s':d[d.duration<=10],'duration_gt10s':d[d.duration>10]}
        # Human annotations are read only AFTER inference; hints are not gold categories.
        ann=out/'dialect_annotations.csv'
        if ann.exists():
            labels=pd.read_csv(ann,keep_default_na=False)
            if labels.ref_uid.duplicated().any(): raise ValueError('Duplicate annotation IDs; adjudicate first')
            # Annotation templates also carry duration/kind/sample_uid for human review.
            # Preserve scoring metadata from the validated rows when joining labels.
            merged=d.merge(labels,on='ref_uid',how='left',validate='one_to_one',suffixes=('', '_annotation'))
            for col in ['accent_region','accent_variation','dialect_vocabulary','vi_accented_english','code_switching','noise']:
                if col in merged:
                    merged[col]=merged[col].fillna('unannotated').replace('', 'unannotated')
                    for label,sub in merged.groupby(col,dropna=False): groups[f'{col}={label}']=sub
        for name,sub in groups.items():
            if len(sub): summaries.append(dict(run=rd.name,subset=name,**aggregate(sub)))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'metrics.csv',index=False)
    return summary,scored


def selection_table(summary,scored,out_path,max_latency_p95_ms=None,max_vram_gb=None):
    """Describe eligible measured tradeoffs; never declare a winner without human review."""
    records=[]
    if not summary.empty:
        for row in summary[summary.subset=='all'].to_dict('records'):
            eligible=row['complete'] and row['n_ok']==row['n'] and row['n']>0 and pd.notna(row['wer'])
            reasons=[]
            if not eligible: reasons.append('incomplete_or_errors')
            if max_latency_p95_ms is not None:
                if pd.isna(row['latency_p95_ms']) or row['latency_p95_ms']>max_latency_p95_ms:
                    eligible=False;reasons.append('latency_limit_or_missing_measurement')
            if max_vram_gb is not None:
                peak=row['vram_peak_reserved_bytes']
                if pd.isna(peak) or peak>max_vram_gb*1024**3:
                    eligible=False;reasons.append('vram_limit_or_missing_measurement')
            records.append(dict(**row,eligible=bool(eligible),excluded_reason=';'.join(reasons),human_review_required=True))
    result=pd.DataFrame(records)
    if not result.empty:
        result['pareto_wer_latency']=False
        eligible=result[result.eligible & result.latency_p95_ms.notna()]
        for i,a in eligible.iterrows():
            dominated=any((other.wer<=a.wer and other.latency_p95_ms<=a.latency_p95_ms and
                (other.wer<a.wer or other.latency_p95_ms<a.latency_p95_ms)) for _,other in eligible.iterrows())
            result.loc[i,'pareto_wer_latency']=not dominated
        result=result.sort_values(['eligible','wer','latency_p95_ms'],ascending=[False,True,True])
    result.to_csv(out_path,index=False)
    return result

def paired_bootstrap(scored,out_path,n_boot=2000,seed=2026):
    """Cluster paired bootstrap by conversation, conditional on fixed evaluation subset."""
    rng=np.random.default_rng(seed);names=sorted(scored);results=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            x,y=scored[a],scored[b]
            if not set(x.ref_uid)==set(y.ref_uid) or x.status.eq('missing').any() or y.status.eq('missing').any(): continue
            if not x.status.eq('ok').any() or not y.status.eq('ok').any(): continue
            merged=x[['ref_uid','sample_uid','word_edits','n_words']].merge(y[['ref_uid','word_edits']],on='ref_uid',suffixes=('_a','_b'),validate='one_to_one')
            groups=merged.groupby('sample_uid')[['word_edits_a','word_edits_b','n_words']].sum().to_numpy()
            if len(groups)<2: continue
            deltas=[]
            for _ in range(n_boot):
                sampled=groups[rng.integers(0,len(groups),len(groups))].sum(axis=0)
                deltas.append((sampled[0]-sampled[1])/sampled[2])
            results.append(dict(a=a,b=b,n_conversations=len(groups),delta_wer_a_minus_b=float((groups[:,0].sum()-groups[:,1].sum())/groups[:,2].sum()),
                                ci95_low=float(np.quantile(deltas,.025)),ci95_high=float(np.quantile(deltas,.975)),n_boot=n_boot))
    pd.DataFrame(results,columns=['a','b','n_conversations','delta_wer_a_minus_b','ci95_low','ci95_high','n_boot']).to_csv(out_path,index=False)
    return results

def save_template_preserving_rows(new,path):
    if path.exists():
        previous=pd.read_csv(path,keep_default_na=False)
        if previous.ref_uid.duplicated().any(): raise ValueError('Duplicate annotation IDs')
        new=pd.concat([previous,new[~new.ref_uid.isin(previous.ref_uid)]],ignore_index=True)
    new.to_csv(path,index=False)

def annotation_templates(rows,out_dir,seed=2026):
    out=Path(out_dir)
    ann=rows[['ref_uid','sample_uid','kind','path','duration']].copy()
    ann['selection_reason']='fixed_benchmark' if len(rows)<=100 else 'fixed_validation'
    for col in ['annotator_id','accent_region','accent_variation','dialect_vocabulary','vi_accented_english','code_switching','noise','reference_problem','notes']:
        ann[col]=''
    save_template_preserving_rows(ann, out/'dialect_annotations.csv')
    gold=rows[['ref_uid','reference']].copy()
    for cat in ['person','money','time','location','device','negation']: gold[cat+'_spans_json']=''
    save_template_preserving_rows(gold, out/'critical_gold_annotation.csv')

def make_review(scored,out_dir):
    records=[]
    for name,d in scored.items():
        for row in d.itertuples():
            r=dict(model=name,ref_uid=row.ref_uid,status=row.status,reference=row.reference,hypothesis=row.text_model)
            for cat in ['person','money','time','location','device','negation']:r[cat+'_judgment']=''
            r['notes']='';records.append(r)
    # Never overwrite filled human review.
    path=Path(out_dir)/'critical_model_review_template.csv'
    new=pd.DataFrame(records,columns=['model','ref_uid','status','reference','hypothesis',
        'person_judgment','money_judgment','time_judgment','location_judgment','device_judgment','negation_judgment','notes'])
    if path.exists():
        old=pd.read_csv(path,keep_default_na=False)
        keys=['model','ref_uid']
        if old.duplicated(keys).any(): raise ValueError('Duplicate model review IDs')
        # Preserve the reviewed hypothesis and judgments, including after retry.
        # A changed hypothesis needs explicit human review rather than silent replacement.
        seen=set(map(tuple,old[keys].to_numpy()))
        new=pd.concat([old,new[[tuple(k) not in seen for k in new[keys].to_numpy()]]],ignore_index=True)
    new.to_csv(path,index=False)

def package_outputs(out_dir,zip_base):
    """Export comparison artifacts only; standalone and usable after interrupted runs."""
    from pathlib import Path
    import hashlib, json, zipfile
    root=Path(out_dir).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    target=Path(str(zip_base)+'.zip').resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    shared={'runtime.json','selection.json','benchmark_request.json','benchmark_source.py','input_provenance.json',
            'scoring_patch.json','scoring_source.py',
            'metrics.csv','paired_bootstrap.csv','comparison_status.csv','model_selection.csv','dependency_check.txt',
            'dialect_annotations.csv','critical_gold_annotation.csv',
            'critical_model_review_template.csv','model_errors.jsonl'}
    run_files={'config.json','model_config.json','generation_config.json',
               'predictions.jsonl','per_audio.csv','recovery.jsonl'}
    selected=[]
    for p in sorted(root.iterdir()):
        if p.is_symlink(): continue
        if p.is_file() and (p.name in shared or
                (p.name.startswith('selection_') and p.suffix=='.csv') or
                p.name.endswith('_checkpoint.json')):
            selected.append(p)
        elif p.is_dir() and not p.name.endswith('_environment') and (p/'config.json').is_file():
            for f in sorted(p.iterdir()):
                if f.is_symlink() or not f.is_file(): continue
                if f.name in run_files or (f.name.startswith(('session_','warmup_')) and f.suffix=='.json') or f.name.startswith('predictions.jsonl.torn-'):
                    selected.append(f)
    entries=[]
    tmp=target.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for f in selected:
                relative=f.relative_to(root).as_posix()
                data=f.read_bytes()
                entries.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                archive.writestr(relative,data)
            manifest=dict(schema_version=1,private=True,files=entries,
                missing_summary_files=[n for n in ['benchmark_request.json','metrics.csv','comparison_status.csv'] if not (root/n).is_file()],
                note='May contain partial runs or load failures. Check comparison_status.csv, n_ok/n_missing and model_errors.jsonl. Contains private transcripts; no original audio or weights exported.')
            archive.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        tmp.replace(target)
    finally:
        if tmp.exists(): tmp.unlink()
    return str(target)
