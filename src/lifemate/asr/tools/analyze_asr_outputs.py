"""Independent, offline review of exported LifeMate ASR results.

Never executes supplied notebook/source, modifies predictions, reads environment
entries, runs inference, or sends data to a service. CSVs contain numeric evidence
and IDs only; raw transcripts remain in the original private archives.
"""
import argparse
import ast
from collections import Counter
import hashlib
import io
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

NORMALIZATION='nfc_lower_punct_symbol_to_space_keep_accents_digits_v1'
PRIMARY=['phowhisper_medium','whisper_turbo','qwen3_asr_06b','qwen3_asr_17b','wav2vec2_vi']

def normalize(text):
    value=unicodedata.normalize('NFC',str(text)).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith(('P','S')) else c for c in value).split())

def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def blocked(name):
    parts=PurePosixPath(str(name).replace('\\','/')).parts
    return any(p.endswith('_environment') for p in parts) or 'state.json' in parts or '..' in parts

class Archive:
    def __init__(self,path):
        self.path=Path(path);self.z=zipfile.ZipFile(self.path)
        self.names=[n for n in self.z.namelist() if not blocked(n)]
        if len(self.names)!=len(set(self.names)): raise ValueError('Duplicate ZIP members')
    def read(self,name):
        if blocked(name) or name not in self.names: raise ValueError('Blocked or missing archive entry: '+name)
        return self.z.read(name)
    def json(self,name): return json.loads(self.read(name))
    def csv(self,name): return pd.read_csv(io.BytesIO(self.read(name)),keep_default_na=False)

def recompute(per,records):
    if per.ref_uid.duplicated().any(): raise ValueError('Duplicate per-audio IDs')
    latest={x['ref_uid']:x for x in records}
    output=[]
    for row in per.itertuples():
        prediction=latest.get(row.ref_uid)
        status=prediction['status'] if prediction else 'missing'
        ref=normalize(row.reference)
        if not ref: raise ValueError('Empty normalized reference: '+row.ref_uid)
        raw=prediction.get('text_model','') if prediction else ''
        hyp=normalize(raw) if status=='ok' else ''
        item=dict(ref_uid=row.ref_uid,sample_uid=row.sample_uid,kind=row.kind,duration_s=float(row.duration),status=status,
            n_words=len(ref.split()),word_edits=Levenshtein.distance(ref.split(),hyp.split()),
            n_chars=len(ref.replace(' ','')),char_edits=Levenshtein.distance(ref.replace(' ',''),hyp.replace(' ','')))
        for col in ['latency_ms','model_ms','rss_peak_bytes','vram_peak_allocated_bytes','vram_peak_reserved_bytes']:
            value=prediction.get(col) if prediction else None
            item[col]=np.nan if value is None else float(value)
        output.append(item)
    return pd.DataFrame(output)

def aggregate(frame):
    ok=frame[frame.status=='ok'];timed=ok[ok.latency_ms.notna()]
    n=len(frame)
    result=dict(n=n,n_conversations=frame.sample_uid.nunique(),n_ok=len(ok),
        n_error=int(frame.status.eq('error').sum()),n_missing=int(frame.status.eq('missing').sum()),
        failure_rate=float(frame.status.ne('ok').mean()),word_edits=int(frame.word_edits.sum()),
        n_words=int(frame.n_words.sum()),char_edits=int(frame.char_edits.sum()),n_chars=int(frame.n_chars.sum()),
        wer=float(frame.word_edits.sum()/frame.n_words.sum()) if len(ok) and frame.n_words.sum() else None,
        cer_no_spaces=float(frame.char_edits.sum()/frame.n_chars.sum()) if len(ok) and frame.n_chars.sum() else None,
        wer_success_only=float(ok.word_edits.sum()/ok.n_words.sum()) if ok.n_words.sum() else None,
        cer_success_only=float(ok.char_edits.sum()/ok.n_chars.sum()) if ok.n_chars.sum() else None,
        latency_p50_ms=float(timed.latency_ms.median()) if len(timed) else None,
        latency_p95_ms=float(timed.latency_ms.quantile(.95)) if len(timed) else None,
        rtf_weighted=float(timed.latency_ms.sum()/1000/timed.duration_s.sum()) if len(timed) else None)
    for col in ['rss_peak_bytes','vram_peak_allocated_bytes','vram_peak_reserved_bytes']:
        result[col]=float(ok[col].max()) if ok[col].notna().any() else None
    return result

def text_flags(reference,hypothesis):
    ref=normalize(reference);hyp=normalize(hypothesis)
    rt=ref.split();ht=hyp.split()
    scripts=sorted({unicodedata.name(c,'UNKNOWN').split(' ')[0] for c in str(hypothesis)
        if c.isalpha() and not unicodedata.name(c,'').startswith('LATIN')})
    trigrams=Counter(tuple(ht[i:i+3]) for i in range(max(0,len(ht)-2)))
    repeats=max(trigrams.values(),default=0)
    # These are review hints, NOT gold hallucination/truncation/semantic error labels.
    return dict(raw_empty=not str(hypothesis).strip(),non_latin_script=bool(scripts),scripts=','.join(scripts),
        short_text_hint=len(rt)>=10 and len(ht)<len(rt)/2,
        long_text_hint=len(rt)>=4 and len(ht)>max(30,2.5*len(rt)),
        repeated_trigram_hint=len(ht)>=20 and repeats>=4,
        max_trigram_occurrences=repeats,hypothesis_tokens=len(ht),reference_tokens=len(rt))

def paired_statistics(frames,n_boot=10000,seed=2026):
    names=sorted(frames)
    if not names:return []
    first=frames[names[0]].set_index('ref_uid').sort_index()
    group_ids=sorted(first.sample_uid.unique())
    words=first.groupby('sample_uid').n_words.sum().reindex(group_ids).to_numpy()
    chars=first.groupby('sample_uid').n_chars.sum().reindex(group_ids).to_numpy()
    we=[];ce=[]
    for name in names:
        f=frames[name].set_index('ref_uid').sort_index()
        if not f.index.equals(first.index) or not f.sample_uid.equals(first.sample_uid): raise ValueError('Unpaired IDs/groups')
        we.append(f.groupby('sample_uid').word_edits.sum().reindex(group_ids).to_numpy())
        ce.append(f.groupby('sample_uid').char_edits.sum().reindex(group_ids).to_numpy())
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(group_ids),np.full(len(group_ids),1/len(group_ids)),size=n_boot)
    boot_wer=(weights@np.array(we).T)/(weights@words)[:,None]
    boot_cer=(weights@np.array(ce).T)/(weights@chars)[:,None]
    results=[]
    for ia,ib in itertools.combinations(range(len(names)),2):
        a,b=names[ia],names[ib]
        fa=frames[a].set_index('ref_uid').sort_index();fb=frames[b].set_index('ref_uid').sort_index()
        delta_w=boot_wer[:,ia]-boot_wer[:,ib];delta_c=boot_cer[:,ia]-boot_cer[:,ib]
        results.append(dict(a=a,b=b,n_audio=len(first),n_conversations=len(group_ids),n_boot=n_boot,seed=seed,
            delta_wer_pp=100*(fa.word_edits.sum()-fb.word_edits.sum())/fa.n_words.sum(),
            wer_ci95_low_pp=float(np.quantile(delta_w,.025)*100),wer_ci95_high_pp=float(np.quantile(delta_w,.975)*100),
            delta_cer_pp=100*(fa.char_edits.sum()-fb.char_edits.sum())/fa.n_chars.sum(),
            cer_ci95_low_pp=float(np.quantile(delta_c,.025)*100),cer_ci95_high_pp=float(np.quantile(delta_c,.975)*100),
            a_lower_word_edits=int(fa.word_edits.lt(fb.word_edits).sum()),b_lower_word_edits=int(fa.word_edits.gt(fb.word_edits).sum()),
            equal_word_edits=int(fa.word_edits.eq(fb.word_edits).sum()),
            both_word_exact=int((fa.word_edits.eq(0)&fb.word_edits.eq(0)).sum()),
            both_nonzero_word_edits=int((fa.word_edits.gt(0)&fb.word_edits.gt(0)).sum()),
            only_a_word_exact=int((fa.word_edits.eq(0)&fb.word_edits.gt(0)).sum()),
            only_b_word_exact=int((fa.word_edits.gt(0)&fb.word_edits.eq(0)).sum()),
            oracle_wer_diagnostic_only=float(np.minimum(fa.word_edits,fb.word_edits).sum()/fa.n_words.sum()),
            interpretation='exploratory_unadjusted_cluster_CI_not_equivalence_or_deployment_router'))
    return results

def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=json_default)+'\n',encoding='utf-8')

def analyze(benchmark,validation,notebook,output,n_boot=10000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=dict(schema_version=1,normalization=NORMALIZATION,inference_executed=False,
        original_predictions_modified=False,limitations=[
            'Original split/reference CSV files and waveform verification not included in these exports.',
            'No speaker-disjoint proof; no human gold annotations; no audio listening or downstream evaluation.',
            'Development validation used for choosing models; benchmark is a subset, not an independent test.',
            'Single-run timing per validation model, same GPU class; clocks/order/CPU RSS may affect resource interpretation.'],
        archives={},notebook={},checks={})
    raw={};frames={};run_manifest=[];metric_rows=[];per_rows=[];deltas=[];load_rows=[];error_rows=[];diagnostic_rows=[]
    for stage,path in [('benchmark',benchmark),('validation',validation)]:
        ar=Archive(path);req=ar.json('benchmark_request.json');source=ar.read('benchmark_source.py')
        source_hash=hashlib.sha256(source).hexdigest()
        if source_hash!=req['config']['source_sha256']: raise ValueError('Source snapshot does not match request')
        declared=ar.json('EXPORT_MANIFEST.json')
        checked=[]
        for entry in declared['files']:
            if blocked(entry['path']):continue
            data=ar.read(entry['path'])
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Export checksum mismatch: '+entry['path'])
            checked.append(entry['path'])
        status=ar.csv('comparison_status.csv');selection=ar.csv('selection_'+stage+'.csv')
        if selection.ref_uid.duplicated().any():raise ValueError('Duplicate selection IDs')
        if selection.ref_uid.tolist()!=req['selected_ref_uids']:raise ValueError('Request/selection mismatch')
        expected=100 if stage=='benchmark' else 991
        if len(selection)!=expected:raise ValueError('Unexpected evaluation size')
        metric_original=ar.csv('metrics.csv');ann=ar.csv('dialect_annotations.csv')
        if ann.ref_uid.duplicated().any(): raise ValueError('Duplicate annotation IDs')
        annotation_fields=['accent_region','accent_variation','dialect_vocabulary','vi_accented_english','code_switching','noise','reference_problem']
        annotation_fill={col:int(ann[col].astype(str).str.strip().ne('').sum()) for col in annotation_fields if col in ann}
        for filename in ['critical_gold_annotation.csv','critical_model_review_template.csv']:
            table=ar.csv(filename)
            for col in table:
                if col.endswith(('_spans_json','_judgment')) or col=='notes':
                    annotation_fill[filename+':'+col]=int(table[col].astype(str).str.strip().ne('').sum())
        inactive=[n.rsplit('/',1)[0] for n in ar.names if n.endswith('/config.json') and n.rsplit('/',1)[0] not in set(status.run)]
        stage_audit=dict(path=str(ar.path.resolve()),sha256=sha(ar.path),allowed_members=len(ar.names),
            checked_manifest_files=len(checked),active_models=len(status),inactive_run_directories=inactive,
            request=req,source_sha256=source_hash,annotation_nonblank=annotation_fill,
            comparisons_to_source_metrics=[],runs=[],runtime=ar.json('runtime.json'),input_provenance=ar.json('input_provenance.json'))
        if 'scoring_patch.json' in ar.names:stage_audit['leftover_scoring_patch']=ar.json('scoring_patch.json')
        raw[stage]={};frames[stage]={}
        canonical=None
        for st in status.itertuples():
            rd=st.run;cfg=ar.json(rd+'/config.json');per=ar.csv(rd+'/per_audio.csv')
            records=[json.loads(line) for line in ar.read(rd+'/predictions.jsonl').splitlines() if line.strip()]
            counts=Counter(x['ref_uid'] for x in records);latest={x['ref_uid']:x for x in records}
            if set(latest)!=set(selection.ref_uid) or per.ref_uid.tolist()!=selection.ref_uid.tolist():raise ValueError('Prediction/selection IDs differ')
            if cfg['model_id']!=st.model_id or cfg['source_sha256']!=source_hash or cfg['normalization']!=NORMALIZATION:raise ValueError('Config mismatch')
            expected_run_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()[:20]
            if not rd.endswith('_'+expected_run_hash):raise ValueError('Run directory fingerprint mismatch')
            meta=per[['ref_uid','sample_uid','kind','duration','reference']].set_index('ref_uid').sort_index()
            if canonical is None:canonical=meta
            elif not meta.equals(canonical):raise ValueError('Reference/sample/kind/duration mismatch between models')
            if not per.sample_uid.str.fullmatch(r'train\+public_test/train/train\.jsonl:\d+').all():raise ValueError('Non-train sample provenance')
            if set(per.kind)-{'final','history'}:raise ValueError('Unexpected role')
            session_ids=set()
            for row in per.itertuples():
                pred=latest[row.ref_uid]
                if pred['model_id']!=cfg['model_id'] or pred['revision']!=cfg['revision'] or pred['run_id']!=expected_run_hash:raise ValueError('Stale prediction identity')
                if pred['status']!=row.status or pred['text_model']!=row.text_model:raise ValueError('Journal/per-audio output mismatch')
                if pred.get('text_normalized')!=normalize(pred['text_model']):raise ValueError('Stored normalization mismatch')
                if pred.get('duration_s') is not None and abs(float(pred['duration_s'])-float(row.duration))>1/16000:raise ValueError('Measured duration differs from metadata')
                session_ids.add(str(pred['session_id']))
                diagnostic_rows.append(dict(stage=stage,model=st.model,ref_uid=row.ref_uid,kind=row.kind,status=pred['status'],
                    **text_flags(row.reference,pred['text_model'])))
                if pred['status']!='ok':error_rows.append(dict(stage=stage,model=st.model,run=rd,ref_uid=row.ref_uid,
                    sample_uid=row.sample_uid,kind=row.kind,error_code=pred.get('error_code'),output_token_count=pred.get('output_token_count'),
                    ended_with_eos=pred.get('ended_with_eos'),duration_s=row.duration))
            computed=recompute(per,records)
            for col in ['word_edits','n_words','char_edits','n_chars']:
                if not np.array_equal(computed[col].to_numpy(),per[col].to_numpy(dtype=int)):raise ValueError('Per-audio scoring mismatch: '+col)
            groups={'all':computed,'final':computed[computed.kind=='final'],'history':computed[computed.kind=='history'],
                'duration_le10s':computed[computed.duration_s<=10],'duration_gt10s':computed[computed.duration_s>10]}
            for subset,sub in groups.items():
                if sub.empty:continue
                metrics=aggregate(sub)
                metric_rows.append(dict(stage=stage,model=st.model,model_id=cfg['model_id'],run=rd,primary_model=st.model in PRIMARY,subset=subset,**metrics))
                original=metric_original[(metric_original.run==rd)&(metric_original.subset==subset)]
                if len(original)!=1:raise ValueError('Source summary missing/duplicate')
                for col in ['n','n_ok','n_error','n_missing','failure_rate','wer','cer_no_spaces','wer_success_only','cer_success_only',
                        'latency_p50_ms','latency_p95_ms','rtf_weighted','rss_peak_bytes','vram_peak_allocated_bytes','vram_peak_reserved_bytes']:
                    old=original.iloc[0][col];new=metrics[col]
                    if new is None:
                        if old not in ['',None] and not pd.isna(old):raise ValueError('Source metric null mismatch')
                    elif not np.isclose(float(old),new,rtol=1e-10,atol=1e-9):raise ValueError(f'Source metric mismatch: {stage}/{st.model}/{subset}/{col}')
                stage_audit['comparisons_to_source_metrics'].append(dict(model=st.model,subset=subset,passed=True))
            stat=aggregate(computed)
            if [st.n_expected,st.n_ok,st.n_error,st.n_missing]!=[len(per),stat['n_ok'],stat['n_error'],stat['n_missing']]:raise ValueError('Status table mismatch')
            for session_id in sorted(session_ids):
                session=ar.json(rd+'/session_'+session_id+'.json');warm=ar.json(rd+'/warmup_'+session_id+'.json')
                if session['hardware']!=cfg['runtime']:raise ValueError('Session/runtime mismatch')
                load_rows.append(dict(stage=stage,model=st.model,run=rd,session_id=session_id,
                    n_prediction_records=sum(str(x['session_id'])==session_id for x in records),load_ms=session['load_ms'],
                    load_rss_peak_bytes=session['load_rss_peak_bytes'],warmup_status=warm['status'],warmup_ms=warm['latency_ms']))
            stage_audit['runs'].append(dict(model=st.model,run=rd,n_records=len(records),n_unique_ids=len(latest),
                duplicate_record_ids=[uid for uid,c in counts.items() if c>1],session_ids=sorted(session_ids),
                word_exact=int(computed.word_edits.eq(0).sum()),empty_raw_outputs=sum(not str(x.get('text_model','')).strip() for x in records),
                error_counts=dict(Counter(x['error_code'] for x in records if x['status']!='ok')),
                warning_counts=dict(Counter(w for x in records for w in x.get('warnings',[])))))
            run_manifest.append(dict(stage=stage,model=st.model,run=rd,config=cfg,
                generation_config=ar.json(rd+'/generation_config.json') if rd+'/generation_config.json' in ar.names else None,
                model_config=ar.json(rd+'/model_config.json')))
            raw[stage][st.model]=per.set_index('ref_uid');frames[stage][st.model]=computed
            per_rows.extend(dict(stage=stage,model=st.model,**r) for r in computed.to_dict('records'))
        ann_idx=ann.set_index('ref_uid')
        if set(canonical.index)-set(ann_idx.index):raise ValueError('Annotation ID missing')
        paths=ann_idx.loc[canonical.index,'path']
        if paths.duplicated().any() or not paths.str.contains('/train+public_test/train/audio/',regex=False).all():raise ValueError('Non-train/duplicate audio provenance')
        final_per_group=canonical[canonical.kind=='final'].groupby('sample_uid').size()
        if not final_per_group.eq(1).all() or (stage=='validation' and len(final_per_group)!=canonical.sample_uid.nunique()):raise ValueError('Final/history grouping inconsistent')
        stage_audit['n_conversations']=canonical.sample_uid.nunique()
        stage_audit['role_counts']=canonical.kind.value_counts().to_dict()
        stage_audit['reference_words']=int(frames[stage][status.iloc[0].model].n_words.sum())
        stage_audit['reference_chars_no_spaces']=int(frames[stage][status.iloc[0].model].n_chars.sum())
        stage_audit['duration_s']=float(canonical.duration.sum())
        audit['archives'][stage]=stage_audit
        pairs=paired_statistics(frames[stage],n_boot=n_boot)
        deltas.extend(dict(stage=stage,**x) for x in pairs)
        ar.z.close()
    if set(frames['benchmark'])!=set(frames['validation']):raise ValueError('Different model set between stages')
    overlaps=[]
    benchmark_ids=set(next(iter(raw['benchmark'].values())).index)
    validation_ids=set(next(iter(raw['validation'].values())).index)
    if not benchmark_ids<=validation_ids:raise ValueError('Benchmark not subset of validation')
    for name in raw['validation']:
        bm=raw['benchmark'][name].sort_index();va=raw['validation'][name].loc[bm.index]
        if not bm[['sample_uid','kind','duration','reference']].equals(va[['sample_uid','kind','duration','reference']]):raise ValueError('Benchmark/validation metadata mismatch')
        overlaps.append(dict(model=name,n_overlap=len(bm),same_raw_text=int(bm.text_model.eq(va.text_model).sum()),
            identical_latency_records=int(bm.latency_ms.eq(va.latency_ms).sum())))
        rest=frames['validation'][name][~frames['validation'][name].ref_uid.isin(benchmark_ids)]
        metric_rows.append(dict(stage='validation_without_benchmark_ids',model=name,
            model_id=audit['archives']['validation']['request']['candidates'][list(raw['validation']).index(name)]['model_id'],
            run='',primary_model=name in PRIMARY,subset='all',**aggregate(rest)))
    audit['overlap']=overlaps
    audit['checks']=dict(export_manifest_checksums_valid=True,source_snapshot_matches_request=True,
        run_fingerprints_valid=True,predictions_match_config=True,metrics_independently_recomputed=True,
        same_ids_references_groups_roles_durations_between_models=True,metadata_train_only=True,
        benchmark_subset_validation=True,annotations_are_templates_only=all(not any(x['annotation_nonblank'].values()) for x in audit['archives'].values()),
        original_development_validation_split_independently_rechecked=False)
    local_audit=Path(__file__).resolve().parents[1]/'audit_verification.json'
    if local_audit.is_file():
        previous=json.loads(local_audit.read_text(encoding='utf-8'))
        audit['recorded_cache_hashes_match_previous_audit']=all(x['request']['cache_hashes']==previous['cache_sha256'] for x in audit['archives'].values())
    notebook=Path(notebook)
    n=json.loads(notebook.read_text(encoding='utf-8'))
    audit['notebook']=dict(path=str(notebook.resolve()),sha256=sha(notebook),n_cells=len(n['cells']),code_cells=[])
    for i,c in enumerate(n['cells']):
        if c['cell_type']!='code':continue
        src=''.join(c['source']);tree=ast.parse(src)
        item=dict(index=i,role=c['metadata'].get('lifemate_role'),execution_count=c.get('execution_count'),
            saved_error_types=[o.get('ename') for o in c.get('outputs',[]) if o.get('output_type')=='error'])
        for node in tree.body:
            if isinstance(node,ast.Assign):
                targets=[t.id for t in node.targets if isinstance(t,ast.Name)]
                if 'BENCHMARK_SOURCE' in targets and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                    item['embedded_source_sha256']=hashlib.sha256(node.value.value.encode()).hexdigest()
                if any(x in targets for x in ['STAGE','SELECTED_MODELS']):
                    try:item[targets[0]]=ast.literal_eval(node.value)
                    except ValueError:pass
        audit['notebook']['code_cells'].append(item)
    metric=pd.DataFrame(metric_rows);metric.to_csv(output/'metrics_recomputed.csv',index=False)
    pd.DataFrame(per_rows).to_csv(output/'per_audio_recomputed.csv',index=False)
    pd.DataFrame(deltas).to_csv(output/'paired_comparisons.csv',index=False)
    pd.DataFrame(load_rows).to_csv(output/'load_warmup_sessions.csv',index=False)
    pd.DataFrame(error_rows).to_csv(output/'inference_errors.csv',index=False)
    diagnostics=pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(output/'text_diagnostic_cases.csv',index=False)
    flag_cols=['raw_empty','non_latin_script','short_text_hint','long_text_hint','repeated_trigram_hint']
    totals=diagnostics.groupby(['stage','model'])[flag_cols].sum().reset_index()
    totals.to_csv(output/'text_diagnostics_summary.csv',index=False)
    save_json(output/'run_manifest.json',run_manifest)
    save_json(output/'audit.json',audit)
    # Metadata-only panel chosen with fixed seed, before examining predictions in the review.
    base=next(iter(raw['validation'].values())).reset_index().sort_values('ref_uid')
    rng=np.random.default_rng(2026);panel=[]
    for kind,n_items in [('final',10),('history',22)]:
        candidates=base[base.kind==kind]
        panel.extend(candidates.iloc[np.sort(rng.choice(len(candidates),n_items,replace=False))][['ref_uid','sample_uid','kind','duration']].to_dict('records'))
    pd.DataFrame(panel).assign(selection='metadata_random_role_stratified_seed2026',audio_listened=False).to_csv(output/'review_panel_ids.csv',index=False)
    print('Wrote analysis to',output)
    print(metric[(metric.stage=='validation')&(metric.subset=='all')][['model','n_ok','n_error','wer','cer_no_spaces','latency_p95_ms']].to_string(index=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--notebook',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=10000)
    args=p.parse_args();analyze(args.benchmark,args.validation,args.notebook,args.output,args.bootstrap)

if __name__=='__main__':main()
