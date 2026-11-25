# PRISM research code and evaluation data

The deposit includes the English PSCon evaluation inputs and reference annotations, the exact four study folds, and the saved outputs and scores for PRISM, both baselines, and the reported follow-up analyses.

## Regenerate the reported tables

From the repository root, run the following with Python 3:

```bash
python3 code/reproduce_saved_tables.py
```

The command writes 11 CSVs to `tables/`, one for each of Tables 1–11. It uses the Python standard library and the saved records, including the finalized scores and manual audit labels. No API credentials, model downloads, or product catalog files are needed. The separate catalog requirements below apply to fresh inference.

## Files by manuscript section

| Manuscript result | Location |
| --- | --- |
| Tables 1–2: evaluation partitions and system configurations | [English PSCon conversations](data/PSCon/) and [exact train, validation, and test splits](data/four_folds/). Each fold’s `test_conversations.json` contains the test requests and original PSCon annotations, including reference actions and recommended products where present. The exporter also writes the system configuration matrix. |
| Tables 3–4: four-fold ranking | [PRISM](results/prism/), [single agent](results/single_agent/), and [fixed workflow](results/fixed_workflow/) contain `turns.csv`, `conversations.csv`, and `summary.json` for each fold. See the [per-fold ranking scores](results/revisions/ranking_by_fold.csv) and [four-fold means and standard deviations](results/revisions/ranking_mean_sd.csv). |
| Table 5 and Figure 3: revision outcomes | [Matched outcomes for all 370 reference events](results/revisions/event_level_matched_370.csv) and [aggregate scores](results/revisions/matched_preference_metrics.json). |
| Table 6 and Figure 4: Fold 3 component substitutions | [Three substitution runs](results/fold3_ablations/) and the [full PRISM Fold 3 reference](results/prism/fold3/). |
| Table 7: critic diagnostic | [Event-level scores, summary, and method](results/critic_diagnostic/). |
| Table 8: targeted manual audit | [Labels for all 28 manually audited events](results/manual_audit/advisor_manual_audit_sample_prcr_spvr.csv). |
| Tables 9–10: metadata evidence and revision types | [Event decisions, evidence, and summaries](results/metadata_audit/). |
| Table 11: external catalog | [Per-turn and per-conversation outputs, automatic diagnostics, and run settings](results/external_catalog/). |
| Section 5.9 and Appendix A | For timing, use the PRISM turn records' `latency_ms` column. Sum per conversation, then average within and across folds. Appendix A's four turns are in [PRISM Fold 3](results/prism/fold3/turns.csv) under conversation ID `14271`. |

`code/` contains the runtime, prompts, evaluators, and catalog processing scripts. The [run manifest](run_manifest.json) indexes the reported runs, metrics, recorded configurations, and source hashes. The model identifier is `deepseek-v4-flash` where recorded. The archived settings do not give an immutable provider revision or a complete record of main-run dependency versions and hardware.

The revision outcome rates use the 370-event PRISM reference inventory for coverage and the selected event–follow-up pairs for conditional rates. Product rejections use exact product IDs. Other automated matches use `all-MiniLM-L6-v2` with a cosine threshold of 0.50 and active-match fraction threshold of 0.50. The scorer selects the first logged recommendation after the revision. Retained sets count when logged as recommended. See the event records for the proxy labels and the manual audit for separately reviewed cases.

## Data sources and catalog access

**PSCon.** Zou et al., *PSCon: Product Search Through Conversations*, SIGIR 2025. [Paper and DOI](https://doi.org/10.1145/3726302.3730278) · [Source dataset and CC BY 4.0 terms](https://github.com/JieZouIR/PSCon). This deposit includes the English data and the exact study splits.

**Amazon Reviews 2023.** Obtain public source data from the [project page](https://amazon-reviews-2023.github.io/) or [Hugging Face distribution](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). The maintainers do not assign a license or set downstream usage terms. Pending redistribution clearance, this deposit omits the processed product records for the 22,203-product main catalog and 50,998-product external catalog.

- Main catalog files held locally: `product_base_table.json`, `product_category_map.json`, and `product_base_category_map.json`. The precise source snapshot and full rebuild recipe are unavailable in the supplied records.
- External catalog files held locally: `catalogue_source.json`, `product_base_table.json`, `product_category_map.json`, and `product_base_category_map.json`. See the [source manifest](results/external_catalog/source_manifest.json), [processing manifest](results/external_catalog/external_manifest.json), and [normalization code](code/external_catalog/prepare.py) for the recorded inputs and `external_schema_v1` processing. The pipeline requires the prepared source snapshot. Full inference also requires fold-specific E5 checkpoints and retrieval artifacts, which are outside this deposit, and an authorized API key.
