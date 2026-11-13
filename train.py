"""
SmartResume AI - NER Model Training & Evaluation Pipeline
Trains a resume-specific Named Entity Recognition model using spaCy.
Integrates custom EntityRuler patterns, dictionary matching, and transition-based NER.
Strictly validates token alignment and evaluates on held-out test data.
"""

import os
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import spacy
from spacy.training import Example, offsets_to_biluo_tags
from spacy.util import minibatch, compounding

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, SKILL_KB_PATH
)

MODEL_OUTPUT_DIR = MODELS_DIR / "resume_ner_model"
EVAL_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Loads records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_entity_ruler_patterns() -> List[Dict[str, Any]]:
    """Builds comprehensive resume entity ruler patterns."""
    patterns = []

    # Load canonical skills from knowledge base
    if SKILL_KB_PATH.exists():
        with open(SKILL_KB_PATH, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            for skill in kb_data.get("skills", []):
                canon = skill["canonical_name"]
                patterns.append({"label": "SKILL", "pattern": canon})
                for alias in skill.get("aliases", []):
                    patterns.append({"label": "SKILL", "pattern": alias})

    # Common degrees
    degrees = [
        "Bachelor of Science", "B.S.", "BS", "Bachelor of Arts", "B.A.", "BA",
        "Master of Science", "M.S.", "MS", "Master of Arts", "M.A.", "MA",
        "Master of Business Administration", "MBA", "Doctor of Philosophy", "Ph.D.", "PhD",
        "Associate of Science", "B.Tech", "M.Tech", "B.E.", "M.E.", "BCA", "MCA",
        "Bachelor of Engineering", "Master of Engineering"
    ]
    for deg in degrees:
        patterns.append({"label": "DEGREE", "pattern": deg})

    # Common job titles
    job_titles = [
        "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
        "Backend Developer", "Frontend Developer", "Data Scientist", "Machine Learning Engineer",
        "DevOps Engineer", "Cloud Architect", "Product Manager", "Project Manager",
        "Business Analyst", "QA Engineer", "Database Administrator", "System Administrator",
        "Solutions Architect", "Data Engineer", "AI Researcher", "IT Director", "General Manager"
    ]
    for jt in job_titles:
        patterns.append({"label": "JOB_TITLE", "pattern": jt})

    # Common certifications
    certifications = [
        "AWS Certified Solutions Architect", "AWS Certified Developer", "CISSP",
        "PMP", "Certified Scrum Master", "CSM", "CompTIA Security+", "Google Cloud Certified",
        "Azure Fundamentals", "Azure Solutions Architect", "CKA", "CKAD"
    ]
    for cert in certifications:
        patterns.append({"label": "CERTIFICATION", "pattern": cert})

    return patterns


def get_strictly_valid_spans(doc, text: str, entities: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    """Aligns character spans to valid token boundaries and eliminates overlaps and invalid BILUO tags."""
    punct = set(string.punctuation).union({'•', '–', '—', '“', '”', '’', '‘', '\"', '\''})
    raw_spans = []

    for ent in entities:
        s, e, l = ent["start"], ent["end"], ent["label"]
        sub = text[s:e]
        sub_s = sub.strip(string.punctuation + " \t\r\n")
        if not sub_s or len(sub_s) < 2:
            continue
        st = sub.find(sub_s)
        new_s = s + st
        new_e = new_s + len(sub_s)
        span = doc.char_span(new_s, new_e, label=l, alignment_mode="contract")
        if span is not None and span.text.strip():
            raw_spans.append((span.start_char, span.end_char, span.label_))

    accepted = []
    for sp in sorted(raw_spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        s, e, l = sp
        if any(not (e <= os or s >= oe) for os, oe, _ in accepted):
            continue
        try:
            test_tags = offsets_to_biluo_tags(doc, accepted + [sp])
            if '-' not in test_tags:
                accepted.append(sp)
        except Exception:
            continue

    return sorted(accepted, key=lambda x: x[0])


def evaluate_ner_model(nlp, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes strict entity-level Precision, Recall, F1-Score, and Support on held-out test data."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for item in test_records:
        text = item["text"]
        doc_gold = nlp.make_doc(text)
        gold_spans = get_strictly_valid_spans(doc_gold, text, item.get("entities", []))

        # Model prediction
        doc_pred = nlp(text)
        pred_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc_pred.ents]

        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        for s, e, label in gold_set:
            support[label] += 1
            if (s, e, label) in pred_set:
                tp[label] += 1
            else:
                fn[label] += 1

        for s, e, label in pred_set:
            if (s, e, label) not in gold_set:
                fp[label] += 1

    per_entity_metrics = {}
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    total_support = sum(support.values())

    all_labels = sorted(list(set(list(support.keys()) + list(tp.keys()) + list(fp.keys()))))

    for label in all_labels:
        c_tp = tp[label]
        c_fp = fp[label]
        c_fn = fn[label]
        c_supp = support[label]

        prec = (c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
        rec = (c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_entity_metrics[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": c_supp,
            "true_positives": c_tp,
            "false_positives": c_fp,
            "false_negatives": c_fn
        }

    overall_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "overall": {
            "precision": round(overall_prec, 4),
            "recall": round(overall_rec, 4),
            "f1_score": round(overall_f1, 4),
            "support": total_support,
            "total_evaluated_resumes": len(test_records)
        },
        "per_entity": per_entity_metrics
    }


def train_model(epochs: int = 6, max_candidates: int = 150):
    """Executes the training and evaluation pipeline."""
    print("=================================================================")
    print(" SmartResume AI - Model Training & Evaluation")
    print("=================================================================")

    train_records = load_jsonl(PROCESSED_DATA_DIR / "train.jsonl")
    val_records = load_jsonl(PROCESSED_DATA_DIR / "validation.jsonl")
    test_records = load_jsonl(PROCESSED_DATA_DIR / "test.jsonl")

    print(f"[+] Loaded splits -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # Initialize English NLP pipeline with NER
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    # Step 1: Collect candidate examples and labels
    print("[*] Filtering and compiling candidate examples...")
    candidates = []
    all_labels = set()

    for item in train_records[:max_candidates]:
        doc = nlp.make_doc(item["text"])
        spans = get_strictly_valid_spans(doc, item["text"], item.get("entities", []))
        if spans:
            ex = Example.from_dict(doc, {"entities": spans})
            candidates.append(ex)
            for _, _, l in spans:
                all_labels.add(l)

    for l in sorted(all_labels):
        ner.add_label(l)
    print(f"[+] Registered NER labels ({len(all_labels)}): {sorted(list(all_labels))}")

    # Initialize model on all labels
    opt = nlp.initialize(get_examples=lambda: candidates)

    # Step 2: Screen examples to keep those fully compatible with transition oracle
    verified_examples = []
    for ex in candidates:
        try:
            nlp.update([ex], sgd=opt)
            verified_examples.append(ex)
        except Exception:
            continue

    print(f"[+] Verified {len(verified_examples)} clean training examples for transition NER")

    if not verified_examples:
        raise ValueError("No verified training examples available!")

    # Step 3: Train for N epochs
    print(f"[*] Training transition NER for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        random.shuffle(verified_examples)
        losses = {}
        for ex in verified_examples:
            try:
                nlp.update([ex], drop=0.2, losses=losses, sgd=opt)
            except Exception:
                continue

        epoch_loss = losses.get("ner", 0.0)
        print(f"  Epoch {epoch:2d}/{epochs} - NER Loss: {epoch_loss:.4f}")

    # Step 4: Add EntityRuler patterns to create a comprehensive hybrid model
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
    patterns = build_entity_ruler_patterns()
    ruler.add_patterns(patterns)
    print(f"[+] Added {len(patterns)} patterns to EntityRuler (hybrid pipeline)")

    # Save trained model to disk
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_OUTPUT_DIR)
    print(f"[+] Model successfully saved to {MODEL_OUTPUT_DIR}")

    # Evaluate strictly on held-out test records
    print(f"[*] Running strict evaluation on held-out test data ({len(test_records)} unseen resumes)...")
    results = evaluate_ner_model(nlp, test_records)

    print("\n--- TEST EVALUATION SUMMARY ---")
    print(f"Overall Precision : {results['overall']['precision'] * 100:.2f}%")
    print(f"Overall Recall    : {results['overall']['recall'] * 100:.2f}%")
    print(f"Overall F1-Score  : {results['overall']['f1_score'] * 100:.2f}%")
    print(f"Total Support     : {results['overall']['support']} entities")
    print("\nPer-Entity Performance:")
    for lbl, m in results["per_entity"].items():
        print(f"  {lbl:15s} | Prec: {m['precision']*100:5.1f}% | Rec: {m['recall']*100:5.1f}% | F1: {m['f1_score']*100:5.1f}% | Support: {m['support']}")

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved detailed evaluation metrics to {EVAL_RESULTS_PATH}")
    print("[✓] Model training and evaluation completed successfully!")


if __name__ == "__main__":
    train_model(epochs=6, max_candidates=150)
