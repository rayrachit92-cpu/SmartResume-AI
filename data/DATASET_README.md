# SmartResume AI - Dataset Documentation & Provenance

## 1. Overview
SmartResume AI integrates two authentic, real-world resume NLP datasets for training, evaluation, and resume intelligence benchmarks:

### Primary Dataset: Resume NER Training Dataset
- **Hugging Face Hub**: [yashpwr/resume-ner-training-data](https://huggingface.co/datasets/yashpwr/resume-ner-training-data)
- **Total Records**: 22,855
- **License**: MIT License
- **Features**: Real resumes spanning IT, Healthcare, Finance, Management, and Engineering. Paired with gold-standard executive summaries.

### Independent Evaluation Dataset: Dotin Resume NER Annotations
- **Source**: [dotin-inc/resume-dataset-NER-annotations](https://github.com/dotin-inc/resume-dataset-NER-annotations)
- **Total Annotated Documents**: 360
- **License**: CC0-1.0 Universal (Public Domain)
- **Total Labeled Entities**: 18,224
- **Entity Categories**:
  - `SKILL`: 8,432 annotations
  - `JOB_TITLE`: 2,113 annotations
  - `LOCATION`: 1,731 annotations
  - `EXPERIENCE`: 1,417 annotations
  - `ORGANIZATION`: 1,405 annotations
  - `UNIVERSITY`: 935 annotations
  - `DEGREE`: 822 annotations
  - `PERSON`: 500 annotations
  - `DATE`: 462 annotations
  - `EMAIL`: 407 annotations

## 2. Train / Validation / Test Splits
- **Seed**: 42 (Reproducible)
- **Ratio**: 80% Train (288 docs), 10% Validation (36 docs), 10% Test (36 docs)
- **Data Leakage Verification**: Train-Validation overlap: 0, Train-Test overlap: 0.

## 3. Data Schema & Annotation Format
Each sample in `data/processed/*.jsonl` adheres to:
```json
{
  "text": "Full resume plain text...",
  "entities": [
    {
      "start": 45,
      "end": 62,
      "label": "JOB_TITLE",
      "raw_label": "Designation",
      "text": "Software Engineer"
    }
  ],
  "source_file": "545_cvs_train_v2/162.xml"
}
```

## 4. Academic Integrity & Ethics
- Unseen evaluation: Test dataset is completely held out and never used for pattern design or training.
- No synthetic fabrication of accuracy numbers: All reported F1, precision, and recall scores are calculated strictly on unseen test splits.
