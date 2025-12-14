#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="sra_cities.tsv"
OUT_DIR="extra"
N_READS=100000

mkdir -p "$OUT_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Brak pliku $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -x "./sample_fasta.py" ]]; then
  echo "sample_fasta.py nie jest wykonywalny albo nie istnieje" >&2
  exit 1
fi

EXTRA_TEST_DATA_TSV="${OUT_DIR}/extra_test_data.tsv"
EXTRA_GT_TSV="${OUT_DIR}/extra_test_ground_truth.tsv"

# Nadpisujemy poprzednie pliki
echo -e "fasta_file" > "$EXTRA_TEST_DATA_TSV"
echo -e "fasta_file\tgeo_loc_name\tlat_lon\ttrain_line" > "$EXTRA_GT_TSV"

# Pomijamy nagłówek
tail -n +2 "$CONFIG_FILE" | while IFS=$'\t' read -r CITY SRR N_SUBSETS; do
  if [[ -z "$CITY" || -z "$SRR" || -z "$N_SUBSETS" ]]; then
    echo "Pomijam linię: CITY='$CITY' SRR='$SRR' N_SUBSETS='$N_SUBSETS'" >&2
    continue
  fi

  FULL_FASTA_GZ="${SRR}.fasta.gz"
  if [[ ! -f "$FULL_FASTA_GZ" ]]; then
    echo "Brak pliku ${FULL_FASTA_GZ}, pomijam $SRR ($CITY)" >&2
    continue
  fi

  echo "=== Miasto=$CITY, SRR=$SRR, n_subsets=$N_SUBSETS ==="

  # W nazwach plików lepiej bez spacji
  CITY_SAFE="${CITY// /_}"

  for ((k=1; k<=N_SUBSETS; k++)); do
    OUT_FASTA_GZ="${OUT_DIR}/${CITY_SAFE}__${SRR}__subset${k}.fasta.gz"
    if [[ -f "$OUT_FASTA_GZ" ]]; then
      echo "  Subset $OUT_FASTA_GZ już istnieje, pomijam sampling."
    else
      echo "  Losuję subset ${k}/${N_SUBSETS} z ${FULL_FASTA_GZ}..."
      ./sample_fasta.py "$FULL_FASTA_GZ" "$OUT_FASTA_GZ" "$N_READS" "$k"
    fi

    FASTA_PATH="$OUT_FASTA_GZ"

    echo "$FASTA_PATH" >> "$EXTRA_TEST_DATA_TSV"

    echo -e "${FASTA_PATH}\t${CITY}\tmissing\textra_${CITY_SAFE}" >> "$EXTRA_GT_TSV"
  done

done
