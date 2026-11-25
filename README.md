# PRISM research code and evaluation data

The deposit includes the English PSCon evaluation inputs and reference annotations, the exact four study folds, and the saved outputs and scores for PRISM, both baselines, and the reported follow-up analyses.

## Regenerate the reported tables

From the repository root, run the following with Python 3:

```bash
python3 code/reproduce_saved_tables.py
```

The command writes 11 CSVs to `tables/`, one for each of IJIM Tables 2–12. Table 1 summarizes related work in the manuscript. If you already generated CSVs with an earlier release, move the old `tables/` directory aside before running the command. It uses the Python standard library and the saved records, including the finalized scores and manual audit labels. No API credentials, model downloads, or product catalog files are needed. The separate catalog requirements below apply to fresh inference.

## Files by manuscript section

| Manuscript result | Location |
| --- | --- |
| Tables 2–3: evaluation partitions and system configurations | [English PSCon conversations](data/PSCon/) and [exact train, validation, and test splits](data/four_folds/). Each fold’s `test_conversations.json` contains the test requests and original PSCon annotations, including reference actions and recommended products where present. The exporter also writes the system configuration matrix. |
| Tables 4–5: four-fold ranking | [PRISM](results/prism/), [single agent](results/single_agent/), and [fixed workflow](results/fixed_workflow/) contain `turns.csv`, `conversations.csv`, and `summary.json` for each fold. See the [per-fold ranking scores](results/revisions/ranking_by_fold.csv) and [four-fold means and standard deviations](results/revisions/ranking_mean_sd.csv). |
| Table 6: revision outcomes | [Matched outcomes for all 370 reference events](results/revisions/event_level_matched_370.csv) and [aggregate scores](results/revisions/matched_preference_metrics.json). |
| Table 7: Fold 3 component substitutions | [Three substitution runs](results/fold3_ablations/) and the [full PRISM Fold 3 reference](results/prism/fold3/). |
| Table 8: critic diagnostic | [Event-level scores, summary, and method](results/critic_diagnostic/). |
| Table 9: targeted manual audit | [Labels for all 28 manually audited events](results/manual_audit/advisor_manual_audit_sample_prcr_spvr.csv). |
| Tables 10–11: metadata evidence and revision types | [Event decisions, evidence, and summaries](results/metadata_audit/). |
| Table 12: external catalog | [Per-turn and per-conversation outputs, automatic diagnostics, and run settings](results/external_catalog/). |
| Section 5.9 and Appendix A | For timing, use the PRISM turn records' `latency_ms` column. Sum per conversation, then average within and across folds. Table A.1’s four turns are in [PRISM Fold 3](results/prism/fold3/turns.csv) under conversation ID `14271`. |

`code/` contains the runtime, prompts, evaluators, and catalog processing scripts. The [run manifest](run_manifest.json) indexes the reported runs, metrics, recorded configurations, and source hashes. The model identifier is `deepseek-v4-flash` where recorded. The archived settings do not give an immutable provider revision or a complete record of main-run dependency versions and hardware.

The revision outcome rates use the 370-event PRISM reference inventory for coverage and the selected event–follow-up pairs for conditional rates. Product rejections use exact product IDs. Other automated matches use `all-MiniLM-L6-v2` with a cosine threshold of 0.50 and active-match fraction threshold of 0.50. The scorer selects the first logged recommendation after the revision. Retained sets count when logged as recommended. See the event records for the proxy labels and the manual audit for separately reviewed cases.

## Data sources and catalog access

**PSCon.** Zou et al., *PSCon: Product Search Through Conversations*, SIGIR 2025. [Paper and DOI](https://doi.org/10.1145/3726302.3730278) · [Source dataset and CC BY 4.0 terms](https://github.com/JieZouIR/PSCon). This deposit includes the English data and the exact study splits.

### Main catalog (22,203 products)

The main product table can be rebuilt byte for byte from the [English PSCon knowledge graph at commit `a7f34236`](https://github.com/JieZouIR/PSCon/blob/a7f34236b86108ee7af78ea6d9a408aca5782af0/dataset/knowledgeGraph_en.txt). Download that file, then run the [rebuild script](code/scripts/rebuild_main_catalog.py) from the repository root:

```bash
python3 code/scripts/rebuild_main_catalog.py /path/to/knowledgeGraph_en.txt /path/to/product_base_table.json
```

The rebuild parses product triples, normalizes whitespace, excludes dialogue feedback edges, and retains the last nonempty value for repeated attributes. The resulting table has SHA-256 `59aaf35f9fa32ec84a0d91190571b0b0caf3947801cba1b651692bc12271b26d`, which matches the archived table and all four original PRISM run manifests.

The original category map survives locally as two identical files, `product_category_map.json` and `product_base_category_map.json`. Its SHA-256 is `8d2cb6c3ffa06d18069465b83c50d99ef667ba21b28acc3280e8448fd4cc22c9`. The original category assignment procedure has not been recovered. The archived July 10, 2026 retrieval build used this catalog with BM25 and a normalized FAISS index built using `all-MiniLM-L6-v2`. Exact regeneration of the original retrieval text and indexes has not been verified.

### External catalog (50,998 products)

The public Amazon Reviews 2023 source is available from the [project page](https://amazon-reviews-2023.github.io/) and [Hugging Face distribution](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). The maintainers do not assign a license or set downstream usage terms. Pending redistribution clearance, the external files `catalogue_source.json`, `product_base_table.json`, `product_category_map.json`, and `product_base_category_map.json` are held locally. See the [source manifest](results/external_catalog/source_manifest.json), [processing manifest](results/external_catalog/external_manifest.json), and [normalization code](code/external_catalog/prepare.py) for the prepared inputs and `external_schema_v1` processing.

Fresh inference also requires fold-specific E5 checkpoints, retrieval artifacts, and an authorized API key. The saved table exports need none of these files.
