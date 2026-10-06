# Business Entity Resolution – Amazon ML Challenge 2026

Match every **Source 1** business record (the de-duplicated reference) to all **Source 2 /
Source 3** records of the same real-world business. Score: macro F0.5 per Source 1 entity,
singletons included. **Leaderboard score of this solution: 0.985602.**

**Pipeline:**
1. text normalisation with dictionaries learned from the training pairs
2. fine-tuned multilingual bi-encoder
3. HNSW approximate nearest-neighbour blocking per country label
4. learned stage-1 prefilter, giving the final candidate set (5.98 candidates per S1 on test)
5. three cross-encoder scores + 59 hand-made pair features
6. LightGBM matcher
7. stage-2 cluster-context re-scorer
8. exclusive assignment + expected-F0.5 set selection

The methodology is described in detail in `../../Documentation_template.md`.

---

## 1. How to reproduce

```bash
# 1) environment: Python 3.14, CUDA GPU (developed on 1x RTX A6000 48 GB, 12-core Xeon, 125 GB RAM)
pip install -r requirements.txt

# 2) point the code to the challenge data (the folder that contains train/ and test/)
export ER_DATA_DIR=/path/to/student_resource/dataset

# 3) run everything: data -> output/matching_results.tsv + output/candidate_pairs.tsv
PY=python bash run_all.sh

# resume from any step
PY=python bash run_all.sh features
```

The first run downloads `intfloat/multilingual-e5-small` (MIT licence, 118M parameters)
from the Hugging Face hub. It is the only pretrained model used.

| variable | default | meaning |
|---|---|---|
| `ER_DATA_DIR` | – (set it) | challenge `dataset/` folder containing `train/` and `test/` |
| `ER_WORK_DIR` | `./work` | intermediate artefacts (parquet, embeddings, models; ~60 GB) |
| `ER_OUTPUT_DIR` | `./output` | `matching_results.tsv`, `candidate_pairs.tsv` |
| `ER_OMP_THREADS` | cores − 4 | threads for LightGBM / FAISS |
| `CE_PAIRS`, `CF_PAIRS` | 1,200,000 / 1,500,000 | training pairs of the cross-encoders |

`predict.py` runs the official `utils/validate_submission.py` at the end.


GPU memory: each model needs only a few GB, but step 7 trains two cross-encoders in
parallel. If one of them fails, it
is retried on its own automatically. Any step can also be resumed with
`bash run_all.sh <step>`.

### Steps (`run_all.sh`)

| # | step | script | what it does | time |
|---|---|---|---|---|
| 1 | prepare | `prepare.py`, `build_dicts.py`, `text_norm.py` | read TSVs, assign folds, learn dictionaries, normalise all records | 3 min |
| 2 | encoder | `encoder.py` | fine-tune the bi-encoder; embed all 24.2M records | 23 + 25 min |
| 3 | densify | `densify.py` | test-like training split (`train_dense`) | 1 min |
| 4 | blocking | `blocking.py` | FAISS HNSW kNN per country, S2/S3 → S1, K=10, cosine-gap pruning | 46 + 22 min |
| 5 | prefilter | `prefilter.py`, `blocking_stats.py` | learned prefilter → **final candidate set** + blocking statistics | 6 min |
| 6 | features | `features.py` | 59 pair features | 11 min |
| 7 | crossencoders | `cross_encoder.py` | train 5 cross-encoders, add scores `ce`, `ce_cf`, `ce_cf2` | ~8.5 h |
| 8 | matcher | `train_matcher.py` | 2 cross-fitted LightGBM matchers, decision-rule search | 5 min |
| 9 | stage2 | `rescore.py` | 2 cross-fitted stage-2 re-scorers | 5 min |
| 10 | predict | `predict.py`, `decision.py`, `io_utils.py` | test scoring, assignment, TSV output, validation | 4 min |



`config.py` holds all paths, folds and hyper-parameters.

---

## 2. Design in brief

* **Folds.** Each S1 cluster (the S1 record + its true S2/S3 records) is hashed into 10
  buckets.
  * Buckets 0–1 learn the dictionaries, the bi-encoder and the first cross-encoder.
  * Buckets 2–5 (fold A) and 6–9 (fold B) train the prefilter, the cross-fitted
    cross-encoders, the matcher and stage 2 by **cross-fitting**: a model trained on
    fold A only ever scores fold B, and vice versa.
  * Every stacked feature and every validation number is therefore out-of-fold.
* **Test-like training split.** The test files have 5.75 S2/S3 records per S1 against
  4.68 in train, with the same ~3.46 true matches per S1, i.e. about twice the distractor
  density.
  * `densify.py` removes 18.73% of the training S1 entities (deterministic hash); their
    records become distractors.
  * The fraction is derived only from record counts of the provided files.
  * All models from blocking onwards are trained and validated on this split.
* **Blocking (candidate generation).**
  * Queries go from the S2/S3 side, since every S2/S3 record has at most one true S1.
  * FAISS HNSW (M=32, efConstruction=200, efSearch=128) runs within each country label.
  * Stage 1: top-10 neighbours, cosine within 0.25 of the best, then the learned
    prefilter (21 cheap features, threshold keeping 99.95% of true pairs).
  * Test: 32.2M kNN pairs → **10.36M final candidate pairs (5.98 per S1)**, with 0.71%
    of the brute-force similarity evaluations.
* **Cross-encoders.**
  * multilingual-e5-small, initialised from the fine-tuned bi-encoder, reading the raw
    `name | address` of both records together.
  * `ce`: trained on buckets 0–1.
  * `ce_cf`: cfA / cfB trained on folds A / B, each scoring only the other fold.
  * `ce_cf2`: cfA2 / cfB2 continue on new pairs of the same fold.
* **Matcher and decision.**
  * LightGBM on 62 features (59 hand-made + 3 cross-encoder scores), then a stage-2
    LightGBM with 34 features (competition, cluster state, sibling agreement).
  * Each S2/S3 record is assigned only to its most probable S1.
  * Per S1, the prefix of candidates that maximises the expected F0.5 is kept, or the
    empty set when "no match" is more likely.
* **Open-set country.** The country label is used only to group blocking; it is never a
  model feature, and nothing is country-specific. France (absent from training) goes
  through the same code.

### Results

| | value |
|---|---|
| **leaderboard macro F0.5** | **0.985602** |
| out-of-fold macro F0.5, test-like split (US + India, 1.43M S1) | 0.98969 (India 0.99103, US 0.98880) |
| final candidates per S1, test (mean / median / p99) | 5.98 / 6 / 13 |
| true pairs kept by the final candidates (test-like split) | 98.92% |
| test matches / predicted singletons | 5,847,352 / 98,786 |

---

## 3. Compliance

* No external data, APIs, geocoding or lookups; only the provided training / test files.
* One pretrained model: `intfloat/multilingual-e5-small` (MIT, 118M parameters). The
  bi-encoder and all five cross-encoders are fine-tuned from it on the training data.
* Libraries: torch, transformers, lightgbm (MIT), faiss-cpu (MIT), rapidfuzz (MIT),
  anyascii (ISC), polars, scikit-learn, tqdm.
* No test labels exist or are used. Unlabelled test records are used only as pipeline
  inputs (embedding, blocking, TF-IDF statistics) and through their record counts for the
  test-like split. Nothing was tuned on the leaderboard.
* Every test S1 entity has exactly one row; matched ids are always a subset of the
  candidate ids.
