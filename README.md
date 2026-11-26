# CB4020: AI in Pharmacovigilance mini-project

**AI-driven adverse event signal screening from PubMed literature: metformin and lactic acidosis.** This is a small, reproducible academic screening pipeline. It is not a validated clinical decision tool and literature co-occurrence is not evidence of causation.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Set an email address for NCBI Entrez before downloading records (NCBI asks users to identify themselves). Optional `NCBI_API_KEY` can be set if you have one.

```bash
export PUBMED_EMAIL="you@example.edu"  # Windows PowerShell: $env:PUBMED_EMAIL="you@example.edu"
python -m src.pipeline
streamlit run app.py
```

The first run searches PubMed and stores the returned real records in `data/raw/raw_data.csv`; reruns use this saved source file. Remove that file if you deliberately want to retrieve a fresh snapshot. Three queries are deduplicated by PMID, with a cap of 200. Only available records are used; records without abstracts remain in the raw file and naturally contribute no sentence-level evidence.

## Outputs

- `data/raw/raw_data.csv`: PMID, title, abstract, year, journal, PubMed URL, UTC retrieval timestamp.
- `data/processed/clean_text.csv`: sentence rows with source PMID and sentence index.
- `data/processed/extracted_entities.csv`: chemical/disease mentions, normalized target terms and extraction method.
- `data/processed/ner_target_validation.csv`: recorded check that BC5CDR recognizes both target entities.
- `data/processed/drug_adr_relations.csv`: target-pair sentence evidence and rule category.
- `data/processed/signal_prioritization.csv`: transparent descriptive ranking summary.
- `data/processed/rule_evaluation.csv`: 12 hand-labeled rule examples.
- `data/processed/relation_review_set.csv`, `relation_validation_results.csv`, `relation_validation_summary.csv`: manually labeled corpus spot-check and descriptive metrics.

## NER and methods

The pipeline uses the scispaCy BC5CDR model `en_ner_bc5cdr_md` to extract chemical and disease entities. Exact-span and entity-label checks identify Metformin as CHEMICAL and lactic acidosis as DISEASE (4/4 target checks pass). A target-term dictionary fallback ensures the selected pair is retained when the model misses a mention; extraction method is recorded per entity.

Relation extraction uses sentence-level contextual cues with scoped negation, uncertainty and supportive patterns; otherwise it records co-occurrence only. Co-occurrence alone is never labeled supportive. The 12 short rule examples passed; a 17-sentence corpus spot-check also passed after manual review. The corpus labels and descriptive metrics were prepared by one reviewer for this project, not through blinded or independent annotation. This small check does not establish general performance. Rules can miss context, scope and cross-sentence relations; categories remain screening labels rather than adjudicated clinical facts.

Priority is a simple display ranking: `3 * min(unique supportive PMIDs, 20) + min(supportive sentences, 40)`. Labels are descriptive thresholds (0–4 limited, 5–19 some, 20+ higher literature support). This arbitrary score is not calibrated or a statistical signal measure. The dashboard compares the literature with the separately supplied FAERS values: 408 reports, ROR 499.446412, PRR 338.044717, EBGM 175.482554, signal label 1. Literature supports a contributory role in some cases, often with independent risk factors, while population studies show mixed findings and context-dependent risk, especially with severe renal impairment. FAERS disproportionality cannot estimate incidence or establish causality, and it should not be compared numerically with publication counts.

The dashboard links representative PubMed findings, shows the NER target checks and relation spot-check, and summarizes limits of comparing spontaneous reports with published studies. FAERS has no exposure denominator and is subject to reporting and data-quality biases; published studies vary in case definitions, populations and design and can be affected by publication bias.

## Reproducibility and limitations

PubMed content changes over time. The raw CSV acts as a retrieval snapshot and records when it was retrieved. Sentence splitting uses NLTK when tokenizer data is available and a punctuation fallback otherwise; negation terms are preserved. The pasted project brief ends partway through its signal prioritization requirements, so the documented score is a clearly specified implementation of the requirements that were provided.

The dashboard includes an evidence-count chart, a NetworkX/Plotly overview, and sentence-level traceability. No literature counts, model performance or clinical conclusions are pre-filled. The supplied FAERS table is displayed separately.
