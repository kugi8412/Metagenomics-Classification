# trees.py
# -*- coding: utf-8 -*-


import gzip
import os
import sys
import time
import gzip
import argparse

import numpy as np
import pandas as pd

from typing import List

from Bio import SeqIO

from evaluate import calculate_auc_roc

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def get_kmers(sequence: str,
              k: int = 6
              ) -> List[str]:
    """ Helper function to extract k-mers.
    Converts SeqRecord to string and creates a list of k-mers.
    """
    seq_str = str(sequence).upper()
    return [seq_str[i: i + k] for i in range(len(seq_str) - k + 1)]


def load_data(tsv_file: str,
              is_train: bool = True
              ):
    """ Loads data based on the TSV file.
    Assumes FASTA files are relative to the TSV file location.
    """
    start_time = time.time()
    base_dir = os.path.dirname(tsv_file)
    
    try:
        df = pd.read_csv(tsv_file, sep='\t')
    except Exception as e:
        print(f"[ERROR]: Reading TSV file {tsv_file}: {e}")
        sys.exit(1)

    seqs = []
    labels = []         # For training
    filenames = []      # For testing
    file_lengths = []   # Reads in a given file for testing
    mode = "TRAINING" if is_train else "TESTING"
    print(f"[{mode}] Loading data from: {tsv_file}")
    
    total_files = df.shape[0]
    
    for index, row in df.iterrows():
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{total_files} files.")

        fname_rel = row.iloc[0]
        file_path = os.path.join(base_dir, fname_rel)
        loc = None
        if is_train:
            if len(row) < 2:
                print(f"[WARNING]: No label in row {index} of training file.")
                continue

            loc = row.iloc[1]

        # Read FASTA file
        if not os.path.exists(file_path):
            if not os.path.exists(file_path):
                print(f"[WARNING]: File {file_path} not found, skipping.")
                if not is_train:
                    filenames.append(fname_rel)
                    file_lengths.append(0)
                continue

        count = 0
        try:
            with gzip.open(file_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seqs.append(str(record.seq))
                    if is_train:
                        labels.append(loc)
                    count += 1
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        if not is_train:
            file_lengths.append(count)
            filenames.append(fname_rel)
    
    elapsed = time.time() - start_time
    print(f"[{mode}] Data loaded in {elapsed:.2f} seconds.")
    return seqs, labels, filenames, file_lengths


def main():
    parser = argparse.ArgumentParser(description="Metagenomic Random Forest Classifier")
    parser.add_argument("training_data", type=str, help="Path to training data TSV")
    parser.add_argument("testing_data", type=str, help="Path to testing data TSV")
    parser.add_argument("output", type=str, help="Path to output TSV")
    parser.add_argument("--k", type=int, default=4, help="K-mer length (default: 4)")
    parser.add_argument("--trees", type=int, default=2, help="Number of trees in forest (default: 2)")
    args = parser.parse_args()
    
    total_start_time = time.time()
    X_train, y_train, _, _ = load_data(args.training_data, is_train=True)
    if not X_train:
        print("[ERROR]: No training data found.")
        sys.exit(1)

    print(f"Number of training sequences: {len(X_train)}")
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    print(f"Classes: {encoder.classes_}")
    model = Pipeline([
        ('kmer_counter', CountVectorizer(analyzer=lambda x: get_kmers(x, k=args.k), binary=False)),
        ('classifier', RandomForestClassifier(n_estimators=args.trees, n_jobs=1, random_state=42, verbose=1))
    ])

    print("Starting training.")
    train_start = time.time()
    model.fit(X_train, y_train_enc)
    train_elapsed = time.time() - train_start
    print(f"Training finished in {train_elapsed:.2f} seconds.")
    X_test, _, test_filenames, test_file_lengths = load_data(args.testing_data, is_train=False)
    if not X_test:
        print("[ERROR]: No testing data found.")
        sys.exit(1)

    print(f"Number of testing sequences: {len(X_test)}")
    print("Starting prediction.")
    pred_start = time.time()
    y_pred_proba = model.predict_proba(X_test)
    pred_elapsed = time.time() - pred_start

    print(f"Prediction finished in {pred_elapsed:.2f} seconds.")
    print("Aggregating results.")
    output_data = []
    current_idx = 0
    all_classes = encoder.classes_
    for i, fname in enumerate(test_filenames):
        length = test_file_lengths[i]
        if length == 0:
            row = [fname] + [0.0] * len(all_classes)
        else:
            file_preds = y_pred_proba[current_idx : current_idx + length]
            avg_pred = np.mean(file_preds, axis=0)
            row = [fname] + list(avg_pred)
            
        output_data.append(row)
        current_idx += length

    cols = ["fasta_file"] + list(all_classes)
    df_out = pd.DataFrame(output_data, columns=cols)
    df_out.to_csv(args.output, sep='\t', index=False)
    total_elapsed = time.time() - total_start_time
    print(f"Results saved to: {args.output}")
    print(f"Total execution time: {total_elapsed/60:.2f} minutes.")

    # Criteria TIME
    total_reads = len(X_train) + len(X_test)
    total_minutes = total_elapsed / 60.0
    if total_reads > 0:
        reads_in_millions = total_reads / 1_000_000.0
        minutes_per_million = total_minutes / reads_in_millions
        
        print("\n" + "="*40)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 40)
        print(f"Total Reads Processed: {total_reads:,}")
        print(f"Total Runtime:         {total_minutes:.2f} min")
        print(f"Processing Rate:       {minutes_per_million:.2f} min / 1M reads")
        print("-" * 40)
        
        if minutes_per_million <= 1.0:
            print("ESTIMATED SCORE: 3 points (Excellent)")
        elif minutes_per_million <= 2.0:
            print("ESTIMATED SCORE: 2 points (Good)")
        elif minutes_per_million <= 5.0:
            print("ESTIMATED SCORE: 1 point (Passable)")
        else:
            print("ESTIMATED SCORE: 0 points (Too slow)")
        print("=" * 40)

    # Optional Evaluation
    if calculate_auc_roc:
        gt_candidate = args.testing_data.replace(".tsv", "_ground_truth.tsv")
        if os.path.exists(gt_candidate):
            print(f"\nFound ground truth: {gt_candidate}. Calculating AUC...")
            try:
                calculate_auc_roc(args.output, gt_candidate)
            except Exception as e:
                print(f"Failed to calculate AUC: {e}")


if __name__ == "__main__":
    main()
