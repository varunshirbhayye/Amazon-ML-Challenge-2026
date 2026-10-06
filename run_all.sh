#!/usr/bin/env bash
# End-to-end pipeline: challenge data -> output/matching_results.tsv + output/candidate_pairs.tsv
#
#   1  prepare         normalise records, learn dictionaries, assign folds
#   2  encoder         fine-tune the multilingual bi-encoder, embed train + test
#   3  densify         test-like training split (test distractor density, from record counts)
#   4  blocking        HNSW approximate kNN per country label (S2/S3 -> S1)
#   5  prefilter       learned stage-1 prefilter -> final candidate set
#   6  features        pair features
#   7  crossencoders   cross-encoder scores ce, ce_cf (cross-fitted), ce_cf2 (cross-fitted)
#   8  matcher         LightGBM matcher, 2-fold cross-fitted
#   9  stage2          stage-2 cluster-context re-scorer
#  10  predict         final matches + candidate file, official validator
#
# Usage:
#   ER_DATA_DIR=/path/to/student_resource/dataset PY=python bash run_all.sh
#   ... bash run_all.sh features          # resume from a step
# Environment (all optional):
#   ER_WORK_DIR / ER_OUTPUT_DIR   intermediate artefacts (default ./work) / outputs (./output)
#   CE_PAIRS, CF_PAIRS            training pairs of the cross-encoders (defaults 1200000, 1500000)
#   DEV_REGEX                     smoke test only: keep clusters whose address matches the regex
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
CE_PAIRS=${CE_PAIRS:-1200000}
CF_PAIRS=${CF_PAIRS:-1500000}
mkdir -p logs
START=${1:-prepare}
STEPS=(prepare encoder densify blocking prefilter features crossencoders matcher stage2 predict)

# train two cross-encoders in parallel; a run that fails (e.g. GPU memory taken by another
# process) is retried on its own, so the step never silently continues with a missing model
train_pair() {  # train_pair "<args model 1>" "<args model 2>" <log1> <log2>
  $PY src/cross_encoder.py train $1 > "$3" 2>&1 &
  local a=$!
  $PY src/cross_encoder.py train $2 > "$4" 2>&1 &
  local b=$!
  if ! wait $a; then echo "retrying alone: $1"; $PY src/cross_encoder.py train $1 > "$3" 2>&1; fi
  if ! wait $b; then echo "retrying alone: $2"; $PY src/cross_encoder.py train $2 > "$4" 2>&1; fi
  grep -a "\[ce\]" "$3" "$4"
}

step() {
  case $1 in
    prepare)
      if [[ -n "${DEV_REGEX:-}" ]]; then $PY src/prepare.py --dev-regex "$DEV_REGEX"; else $PY src/prepare.py; fi ;;
    encoder)
      $PY src/encoder.py train
      $PY src/encoder.py embed --split train
      $PY src/encoder.py embed --split test ;;
    densify)
      $PY src/densify.py ;;
    blocking)
      $PY src/blocking.py --split train_dense
      $PY src/blocking.py --split test ;;
    prefilter)
      $PY src/prefilter.py --split train_dense
      $PY src/prefilter.py --split test ;;
    features)
      $PY src/features.py --split train_dense
      $PY src/features.py --split test ;;
    crossencoders)
      # ce: trained on the encoder-bucket clusters only
      $PY src/cross_encoder.py train --split train_dense --max-pairs "$CE_PAIRS"
      $PY src/cross_encoder.py score --split train_dense --models ce --col ce
      $PY src/cross_encoder.py score --split test --models ce --col ce
      # ce_cf: cfA on fold A, cfB on fold B (in parallel), each scores only the other fold
      train_pair "--buckets A --init ce --out cfA --max-pairs $CF_PAIRS" \
                 "--buckets B --init ce --out cfB --max-pairs $CF_PAIRS" logs/train_cfA.log logs/train_cfB.log
      $PY src/cross_encoder.py score --split train_dense --models cfA,cfB --col ce_cf
      $PY src/cross_encoder.py score --split test --models cfA,cfB --col ce_cf
      # ce_cf2: continue on NEW pairs of the same fold (pairs of the previous round excluded)
      train_pair "--buckets A --init cfA --out cfA2 --max-pairs $CF_PAIRS --exclude-prev $CF_PAIRS --lr 2e-5" \
                 "--buckets B --init cfB --out cfB2 --max-pairs $CF_PAIRS --exclude-prev $CF_PAIRS --lr 2e-5" \
                 logs/train_cfA2.log logs/train_cfB2.log
      $PY src/cross_encoder.py score --split train_dense --models cfA2,cfB2 --col ce_cf2
      $PY src/cross_encoder.py score --split test --models cfA2,cfB2 --col ce_cf2 ;;
    matcher)
      $PY src/train_matcher.py --split train_dense ;;
    stage2)
      $PY src/rescore.py --split train_dense ;;
    predict)
      $PY src/predict.py --stage2 ;;
  esac
}

run=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && run=1
  [[ $run == 0 ]] && continue
  echo "================ $s  ($(date '+%F %T'))"
  step "$s" 2>&1 | tee "logs/$s.log"
done
echo "================ done ($(date '+%F %T'))"
