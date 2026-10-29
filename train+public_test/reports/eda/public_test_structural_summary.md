**Unlabeled distribution audit, not a performance evaluation.** This ran only after training analysis and manifest construction. No public audio transcription, intent/request inspection, pseudo-labeling, evaluation, or tuning took place.

| measure | train | public_test |
| --- | --- | --- |
| Samples | 1545 | 634 |
| Nonempty history samples | 1018 | 419 |
| Mean history turns | 4.519 | 4.098 |
| Mean installed apps | 9.201 | 8.528 |
| Mean permitted tools | 32.528 | 30.659 |
| Median current duration (s) | 6.936 | 7.209 |
| P95 current duration (s) | 11.836 | 13.465 |
| Unique referenced WAVs | 5036 | 1933 |

Public catalog: 36 apps, 122 tools, schema version 2.7.0. Catalog comparison: 59 train-only tools, 34 public-only tools, 0 shared tools with different parameter schemas, 0 shared tools with any definition difference. Full structural differences and distributions are in public_test_structural_summary.json and public_* CSVs.

Catalog differences describe scope; they do not establish model generalization or success. Public context values are never exported. No training partitions or model recommendations were altered in response to this audit.
