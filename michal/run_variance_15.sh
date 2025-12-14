#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="sra_cities.tsv"
TRAIN_TSV="train1_data.tsv"

RUNS=15
SUBSETS_PER_FASTA=3
N_READS=100000

OUT_ROOT="variance_runs"
mkdir -p "$OUT_ROOT"

VARIANCES_TXT="variances.txt"
: > "$VARIANCES_TXT"   # nadpisz

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Brak $CONFIG_FILE" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_TSV" ]]; then
  echo "Brak $TRAIN_TSV" >&2
  exit 1
fi
if [[ ! -x "./sample_fasta.py" ]]; then
  echo "sample_fasta.py nie jest wykonywalny (chmod +x sample_fasta.py)" >&2
  exit 1
fi

echo -e "run_id\tavg_auc" >> "$VARIANCES_TXT"

for run_id in $(seq 1 "$RUNS"); do
  echo "=============================="
  echo "RUN $run_id / $RUNS"
  echo "=============================="

  RUN_DIR="${OUT_ROOT}/run_${run_id}"
  mkdir -p "$RUN_DIR"

  DATA_TSV="${RUN_DIR}/extra_test_data_${run_id}.tsv"
  GT_TSV="${RUN_DIR}/extra_test_ground_truth_${run_id}.tsv"
  PRED_TSV="${RUN_DIR}/variance_${run_id}.tsv"
  EVAL_TXT="${RUN_DIR}/eval_${run_id}.txt"

  echo -e "fasta_file" > "$DATA_TSV"
  echo -e "fasta_file\tgeo_loc_name\tlat_lon\ttrain_line" > "$GT_TSV"

  # pomijamy nagłówek configa
  tail -n +2 "$CONFIG_FILE" | while IFS=$'\t' read -r CITY SRR _N_SUBSETS_UNUSED; do
    [[ -z "${CITY:-}" || -z "${SRR:-}" ]] && continue

    FULL_FASTA_GZ="${SRR}.fasta.gz"
    if [[ ! -f "$FULL_FASTA_GZ" ]]; then
      echo "[WARN] Brak ${FULL_FASTA_GZ}, pomijam SRR=$SRR CITY=$CITY" >&2
      continue
    fi

    CITY_SAFE="${CITY// /_}"

    for subset_idx in $(seq 1 "$SUBSETS_PER_FASTA"); do
      SRR_NUM="${SRR//[^0-9]/}"
      SRR_NUM="${SRR_NUM:-0}"
      seed=$(( run_id*1000003 + subset_idx*10007 + (SRR_NUM % 100000) ))

      OUT_FASTA_GZ="${RUN_DIR}/${CITY_SAFE}__${SRR}__run${run_id}_subset${subset_idx}.fasta.gz"
      if [[ ! -f "$OUT_FASTA_GZ" ]]; then
        echo "  Sampling: $OUT_FASTA_GZ (seed=$seed)"
        ./sample_fasta.py "$FULL_FASTA_GZ" "$OUT_FASTA_GZ" "$N_READS" "$seed"
      fi
      
      bn="$(basename "$OUT_FASTA_GZ")"
      echo "$bn" >> "$DATA_TSV"
      echo -e "${bn}\t${CITY}\tmissing\trun_${run_id}" >> "$GT_TSV"

    done
  done

  echo ">> Classifier..."
  python3 classifier_piotr.py "$TRAIN_TSV" "$DATA_TSV" "$PRED_TSV"

  echo ">> Evaluate..."
  python3 evaluate.py "$PRED_TSV" "$GT_TSV" | tee "$EVAL_TXT"

  avg_auc="$(awk '/Average AUC-ROC across all classes:/{print $NF}' "$EVAL_TXT" | tail -n 1)"

  echo -e "${run_id}\t${avg_auc}" >> "$VARIANCES_TXT"
done

echo
echo "Gotowe."
echo "Wyniki: $VARIANCES_TXT"
echo "Pliki runów: $OUT_ROOT/run_*/"
