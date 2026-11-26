"""Small, reproducible literature screening pipeline for metformin and lactic acidosis."""
from __future__ import annotations

import os
import re
import time
import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from Bio import Entrez

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/raw_data.csv"
PROCESSED = ROOT / "data/processed"
QUERIES = [
    'metformin AND "lactic acidosis"',
    '"metformin-associated lactic acidosis"',
    '"metformin-induced lactic acidosis"',
]
DRUG_RE = re.compile(r"\bmetformin\b", re.I)
ADR_RE = re.compile(r"\blactic\s+acidosis\b", re.I)


def fetch_pubmed(email: str | None = None, max_records: int = 200) -> pd.DataFrame:
    """Fetch real PubMed records using Entrez; requires PUBMED_EMAIL."""
    email = email or os.getenv("PUBMED_EMAIL")
    if not email:
        raise RuntimeError("Set PUBMED_EMAIL to your contact email before querying NCBI.")
    Entrez.email = email
    if os.getenv("NCBI_API_KEY"):
        Entrez.api_key = os.environ["NCBI_API_KEY"]
    ids: list[str] = []
    # Search individual concepts, deduplicate, then cap the combined set.
    for query in QUERIES:
        with Entrez.esearch(db="pubmed", term=query, retmax=max_records, sort="relevance") as h:
            ids.extend(Entrez.read(h)["IdList"])
        time.sleep(0.35)
    ids = list(dict.fromkeys(ids))[:max_records]
    records = []
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        if not batch:
            continue
        with Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml") as h:
            articles = Entrez.read(h)["PubmedArticle"]
        for item in articles:
            med = item["MedlineCitation"]
            article = med["Article"]
            abstract = article.get("Abstract", {}).get("AbstractText", [])
            abstract_text = " ".join(str(x) for x in abstract).strip()
            pubdate = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pubdate.get("Year") or str(pubdate.get("MedlineDate", ""))[:4]
            records.append({
                "pmid": str(med["PMID"]), "title": str(article.get("ArticleTitle", "")),
                "abstract": abstract_text, "year": str(year),
                "journal": str(article.get("Journal", {}).get("Title", "")),
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{med['PMID']}/",
            })
        time.sleep(0.35)
    df = pd.DataFrame(records).drop_duplicates("pmid")
    df["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    return df


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = html.unescape(str(text))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def sentences(text: str) -> list[str]:
    # Sentence tokenizer keeps sentence content (including negation) unchanged.
    try:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except LookupError:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.fillna("").iterrows():
        abstract = clean_text(row.get("abstract", ""))
        title = clean_text(row.get("title", ""))
        alltext = " ".join(x for x in (title, abstract) if x)
        for i, sentence in enumerate(sentences(alltext)):
            rows.append({"pmid": str(row.get("pmid", "")), "title": title,
                         "year": row.get("year", ""), "journal": row.get("journal", ""),
                         "pubmed_url": row.get("pubmed_url", ""), "retrieved_at_utc": row.get("retrieved_at_utc", ""), "sentence_index": i,
                         "sentence": sentence})
    out = pd.DataFrame(rows, columns=["pmid", "title", "year", "journal", "pubmed_url", "retrieved_at_utc", "sentence_index", "sentence"])
    out.to_csv(PROCESSED / "clean_text.csv", index=False)
    return out


def extract_entities(clean: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Use scispaCy BC5CDR if installed; otherwise explicitly use dictionary matching."""
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
    except (ImportError, OSError):
        pass
    validation = []
    if nlp:
        checks = [
            ("Metformin may cause lactic acidosis.", "Metformin", "CHEMICAL"),
            ("Metformin may cause lactic acidosis.", "lactic acidosis", "DISEASE"),
            ("Patients with lactic acidosis were treated with Metformin.", "Metformin", "CHEMICAL"),
            ("Patients with lactic acidosis were treated with Metformin.", "lactic acidosis", "DISEASE"),
        ]
        for sentence, target, label in checks:
            detected = any(ent.text.lower() == target.lower() and ent.label_ == label for ent in nlp(sentence).ents)
            validation.append({"sentence": sentence, "target": target, "expected_label": label,
                               "detected": detected, "validation_method": "manual target-term check"})
    pd.DataFrame(validation, columns=["sentence", "target", "expected_label", "detected", "validation_method"]).to_csv(
        PROCESSED / "ner_target_validation.csv", index=False)
    rows = []
    for row in clean.itertuples(index=False):
        text = row.sentence
        found: list[tuple[str, str, str]] = []
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                typ = "chemical" if ent.label_ == "CHEMICAL" else "disease" if ent.label_ == "DISEASE" else None
                if typ:
                    found.append((ent.text, typ, "scispaCy en_ner_bc5cdr_md"))
        # Target-pair normalization is applied transparently, including when model misses a mention.
        for m in DRUG_RE.finditer(text):
            found.append((m.group(), "chemical", "dictionary target-term fallback"))
        for m in ADR_RE.finditer(text):
            found.append((m.group(), "disease", "dictionary target-term fallback"))
        seen = set()
        for entity, typ, method in found:
            normalized = "Metformin" if typ == "chemical" and DRUG_RE.fullmatch(entity) else "Lactic acidosis" if typ == "disease" and ADR_RE.fullmatch(entity) else entity
            key = (entity.lower(), typ)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"pmid": row.pmid, "sentence": text, "entity_text": entity,
                         "normalized_entity": normalized, "entity_type": typ, "extraction_method": method})
    entities = pd.DataFrame(rows, columns=["pmid", "sentence", "entity_text", "normalized_entity", "entity_type", "extraction_method"])
    entities.to_csv(PROCESSED / "extracted_entities.csv", index=False)
    return entities, "scispaCy BC5CDR with target dictionary fallback" if nlp else "dictionary target-term matching (scispaCy unavailable)"


NEGATED = ["no evidence", "not associated", "not linked", "not related", "without evidence", "absence of", "did not", "was not", "were not", "no association"]
UNCERTAIN = ["possibly", "uncertain", "unclear", "cannot exclude", "may possibly", "might", "could potentially"]
SUPPORTIVE = ["associated", "induced", "caused", "cause", "contribute", "attribut", "risk of", "due to", "related to", "linked to", "case of", "developed"]


def classify_relation(sentence: str) -> tuple[str, str]:
    low = sentence.lower()
    # Negation has precedence over supportive cues. Uncertainty then precedes generic association.
    for phrase in NEGATED:
        if phrase in low:
            return "Negated", f"negation cue: {phrase}"
    for phrase in UNCERTAIN:
        if phrase in low:
            return "Uncertain", f"uncertainty cue: {phrase}"
    for phrase in SUPPORTIVE:
        if phrase in low:
            return "Supportive", f"association cue: {phrase}"
    # Hyphenated compound names are a direct association construction.
    if re.search(r"metformin[- ](?:associated|induced)[- ]lactic acidosis", low):
        return "Supportive", "compound association phrase"
    return "Co-occurrence only", "both target entities present; no relation cue"


def extract_relations(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in clean.itertuples(index=False):
        if DRUG_RE.search(row.sentence) and ADR_RE.search(row.sentence):
            category, rule = classify_relation(row.sentence)
            rows.append({"pmid": row.pmid, "sentence": row.sentence, "drug": "Metformin",
                         "adr": "Lactic acidosis", "relationship_category": category, "evidence_rules": rule})
    cols = ["pmid", "sentence", "drug", "adr", "relationship_category", "evidence_rules"]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(PROCESSED / "drug_adr_relations.csv", index=False)
    return out


def prioritize(relations: pd.DataFrame) -> pd.DataFrame:
    """Transparent descriptive score; not a causal estimate or validated clinical score."""
    supportive = relations[relations.relationship_category == "Supportive"] if not relations.empty else relations
    unique_pubs = supportive.pmid.nunique() if not supportive.empty else 0
    support_sentences = len(supportive)
    # Simple additive points; capped to prevent a few prolific articles dominating.
    score = min(unique_pubs, 20) * 3 + min(support_sentences, 40)
    label = "Higher literature support" if score >= 20 else "Some literature support" if score >= 5 else "Limited literature support"
    result = pd.DataFrame([{"drug": "Metformin", "adr": "Lactic acidosis", "supportive_publications": unique_pubs,
                            "supportive_sentences": support_sentences, "priority_score": score,
                            "priority_label": label, "score_definition": "3*min(unique supportive PMIDs,20)+min(supportive sentences,40)"}])
    result.to_csv(PROCESSED / "signal_prioritization.csv", index=False)
    return result


def evaluate_rule_examples() -> pd.DataFrame:
    """Sanity-check documented cues on small hand-labeled examples, not a gold-standard evaluation."""
    examples = [
        ("Metformin-associated lactic acidosis was observed.", "Supportive"),
        ("Metformin was associated with lactic acidosis.", "Supportive"),
        ("Metformin may contribute to lactic acidosis.", "Supportive"),
        ("Metformin-induced lactic acidosis was reported.", "Supportive"),
        ("Metformin caused lactic acidosis.", "Supportive"),
        ("No evidence of metformin-induced lactic acidosis was found.", "Negated"),
        ("Metformin was not associated with lactic acidosis.", "Negated"),
        ("There is no association between metformin and lactic acidosis.", "Negated"),
        ("No evidence links metformin to lactic acidosis.", "Negated"),
        ("Metformin may possibly increase risk of lactic acidosis.", "Uncertain"),
        ("It is uncertain whether metformin causes lactic acidosis.", "Uncertain"),
        ("Metformin and lactic acidosis were both discussed.", "Co-occurrence only"),
    ]
    from sklearn.metrics import accuracy_score
    rows = []
    for sentence, expected in examples:
        predicted, rule = classify_relation(sentence)
        rows.append({"sentence": sentence, "expected_category": expected,
                     "predicted_category": predicted, "correct": predicted == expected,
                     "evidence_rule": rule})
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / "rule_evaluation.csv", index=False)
    print(f"Rule example sanity check: {accuracy_score(out.expected_category, out.predicted_category):.0%} ({out.correct.sum()}/{len(out)})")
    return out


def run() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        df = fetch_pubmed()
    else:
        df = pd.read_csv(RAW, dtype={"pmid": str}).fillna("")
        if "retrieved_at_utc" not in df:
            retrieved_at = datetime.fromtimestamp(RAW.stat().st_mtime, timezone.utc).isoformat()
            df["retrieved_at_utc"] = retrieved_at
            df.to_csv(RAW, index=False)
    clean = preprocess(df)
    _, method = extract_entities(clean)
    rel = extract_relations(clean)
    prioritize(rel)
    evaluate_rule_examples()
    print(f"Loaded {len(df)} PubMed records; entity method: {method}; relations: {len(rel)}")


if __name__ == "__main__":
    run()
