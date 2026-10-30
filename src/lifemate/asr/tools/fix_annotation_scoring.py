"""Paste into a Kaggle code cell after the failed model cell; do not restart kernel.

Only patch scoring/export. Keep cfg/run IDs and predictions from the current inference.
"""
old="merged=d.merge(labels,on='ref_uid',how='left',validate='one_to_one')"
new="merged=d.merge(labels,on='ref_uid',how='left',validate='one_to_one',suffixes=('', '_annotation'))"
if old not in BENCHMARK_SOURCE:
    raise RuntimeError('Expected old merge was not found; use the newly generated notebook instead.')
patched=BENCHMARK_SOURCE.replace(old,new)
patched=patched.replace("'benchmark_source.py',", "'benchmark_source.py','scoring_source.py','scoring_patch.json',")
exec(compile(patched,'asr_benchmark_scoring_patch.py','exec'))
(OUTPUT_DIR/'scoring_source.py').write_text(patched,encoding='utf-8')
write_json(OUTPUT_DIR/'scoring_patch.json',dict(
    patch='preserve_scoring_columns_when_merging_annotations',
    original_inference_source_sha256=cfg['source_sha256'],
    scoring_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
    inference_rerun=False))
refresh_results()
display(comparison_status)
