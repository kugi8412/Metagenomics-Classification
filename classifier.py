# hashes.py
# -*- coding: utf-8 -*-


import os
import gzip
import argparse

import numpy as np

from typing import Tuple
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import resource
except ImportError:
    resource = None

# Config
DEFAULT_K = 25
DEFAULT_SKETCH_SIZE = 7500
DEFAULT_SEED = 424242
CHUNK_SIZE_KB = 75


def get_pow4_kernel(k: int) -> np.ndarray:
    """ Returns weights [4^(k-1), ..., 1] for packing k-mers into int64.
    """
    return (4 ** np.arange(k - 1, -1, -1)).astype(np.uint64)


def vectorized_hash(arr: np.ndarray) -> np.ndarray:
    """ Invertible integer mixing function (Wang Hash) for randomization.
    # MurmurHash3 64-bit a bit faster but get worse AUC
    key = arr.astype(np.uint64)
    key ^= key >> np.uint64(33)
    key *= np.uint64(0xff51afd7ed558ccd)
    key ^= key >> np.uint64(33)
    key *= np.uint64(0xc4ceb9fe1a85ec53)
    key ^= key >> np.uint64(33)
    """
    key = arr.copy()
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)
    return key


def process_chunk_vectorized(sequence_buffer: str,
                             k: int,
                             pow4_kernel: np.ndarray
                             ) -> np.ndarray:
    """ Vectorized k-mer hashing using numpy.
    Significantly faster than standard Python loops.
    """
    if not sequence_buffer:
        return np.array([], dtype=np.uint64)

    # Map ASCII to Int (A=0, C=1, G=2, T=3, Other=4)
    mapper = np.full(256, 4, dtype=np.int8)
    mapper[ord('A')] = 0
    mapper[ord('a')] = 0
    mapper[ord('C')] = 1
    mapper[ord('c')] = 1
    mapper[ord('G')] = 2
    mapper[ord('g')] = 2
    mapper[ord('T')] = 3
    mapper[ord('t')] = 3
    arr_ascii = np.frombuffer(sequence_buffer.encode('ascii'), dtype=np.uint8)
    arr_int = mapper[arr_ascii]
    
    L = len(arr_int)
    if L < k:
        return np.array([], dtype=np.uint64)

    # Sliding window
    shape = (L - k + 1, k)
    strides = (arr_int.strides[0], arr_int.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr_int, shape=shape, strides=strides)
    window_max = np.max(windows, axis=1)
    valid_indices = window_max < 4

    if not np.any(valid_indices):
        return np.array([], dtype=np.uint64)

    valid_windows = windows[valid_indices].astype(np.uint64)
    packed_fwd = valid_windows.dot(pow4_kernel)
    rc_windows = (3 - valid_windows[:, ::-1])
    packed_rc = rc_windows.dot(pow4_kernel)
    
    # Hash
    packed_canonical = np.minimum(packed_fwd, packed_rc)
    return vectorized_hash(packed_canonical)


def sketch_file_fast(filename: str,
                     k: int,
                     sketch_size: int
                     ) -> Tuple[np.ndarray]:
    """ Reads file in large chunks and sketches using vectorization.
    Returns: (sketch_set, approx_read_count)
    """
    pow4 = get_pow4_kernel(k)
    all_hashes = set()
    try:
        with gzip.open(filename, 'rt') as f:
            overlap = ""
            while True:
                chunk = f.read(CHUNK_SIZE_KB * 1024)
                if not chunk:
                    break

                full_text = overlap + chunk
                if len(chunk) >= k:
                    overlap = full_text[-(k-1):]
                else:
                    overlap = full_text
                clean_text = full_text.replace('\n', '')
                hashes = process_chunk_vectorized(clean_text, k, pow4)
                if len(hashes) > 0:
                    all_hashes.update(hashes)

                    # Save memory
                    if len(all_hashes) > sketch_size * 20:
                         arr = np.array(list(all_hashes), dtype=np.uint64)
                         part = np.partition(arr, sketch_size * 5)
                         all_hashes = set(part[:sketch_size * 5])

    except Exception as e:
        print(f"[ERROR]: Processing {filename}: {e}")

    if not all_hashes:
        return set(), 0

    final_arr = np.array(list(all_hashes), dtype=np.uint64)
    if len(final_arr) > sketch_size:
        final_arr = np.partition(final_arr, sketch_size)[:sketch_size]

    return set(final_arr)


# Logistic Regression
def train_lr(X: np.ndarray,
             y: np.ndarray
             ) -> object:
    """ Trains a Scikit-Learn Logistic Regression pipeline.
    Replaces manual scipy.optimize.minimize. Solver='lbfgs' and
    less regularization (C=10.0) for better convergence.
    """
    model = make_pipeline(
        StandardScaler(), 
        LogisticRegression(
            C=10.0, 
            solver="lbfgs", 
            max_iter=1000, 
            class_weight="balanced",
            random_state=DEFAULT_SEED
        )
    )
    
    model.fit(X, y)
    return model


def predict(model: object, X: np.ndarray) -> np.ndarray:
    """ Wrapper for sklearn prediction to match expected output format.
    Returns probability matrix (n_samples, n_classes).
    """
    return model.predict_proba(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data")
    parser.add_argument("testing_data")
    parser.add_argument("output")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--sketch_size", type=int, default=DEFAULT_SKETCH_SIZE)
    args = parser.parse_args()
    file_map = {}
    base_dir = os.path.dirname(args.training_data)
    with open(args.training_data) as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p)>=2: 
                if p[1] not in file_map: file_map[p[1]] = []
                file_map[p[1]].append(os.path.join(base_dir, p[0]))
    
    labels = sorted(file_map.keys())
    lab2idx = {l: i for i,l in enumerate(labels)}
    ref_sketches = []
    for lab in labels:
        city_hashes = set()
        for fpath in file_map[lab]:
            s = sketch_file_fast(fpath, args.k, args.sketch_size)
            city_hashes.update(s)
            
        final_arr = np.array(list(city_hashes), dtype=np.uint64)

        # Take more sketches
        if len(final_arr) > args.sketch_size * 2:
             final_arr = np.partition(final_arr, args.sketch_size * 2)[:args.sketch_size * 2]
        
        ref_sketches.append(set(final_arr))

    X_train, y_train = [], []
    pow4 = get_pow4_kernel(args.k)
    
    for lab in labels:
        idx = lab2idx[lab]
        for fpath in file_map[lab]:
            try:
                with gzip.open(fpath, 'rt') as f:
                    for _ in range(10):
                        chunk = f.read(CHUNK_SIZE_KB * 1024)
                        if not chunk or len(chunk) < 5000:
                            break

                        hashes = process_chunk_vectorized(chunk.replace('\n',''), args.k, pow4)
                        if len(hashes) > 0:
                            s_sample = set(np.unique(hashes))
                            if len(s_sample) > args.sketch_size:
                                arr = np.array(list(s_sample), dtype=np.uint64)
                                s_sample = set(np.partition(arr, args.sketch_size)[:args.sketch_size])

                            feats = []
                            denom = len(s_sample)
                            if denom > 0:
                                for ref in ref_sketches:
                                    feats.append(len(s_sample & ref) / denom)
                                X_train.append(feats)
                                y_train.append(idx)
            except:
                pass

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if len(X_train) == 0:
        return None

    models = train_lr(X_train, y_train)

    test_files = []
    base_test = os.path.dirname(args.testing_data)
    with open(args.testing_data) as f:
        lines = [x.strip() for x in f if x.strip()]
        test_files = lines[1:] if lines[0].startswith("fasta") else lines
        
    results = []
    for i, fname in enumerate(test_files):
        fpath = os.path.join(base_test, fname)
        s_set = sketch_file_fast(fpath, args.k, args.sketch_size)
        feats = []
        denom = len(s_set)
        if denom > 0:
            for ref in ref_sketches:
                feats.append(len(s_set & ref)/denom)
        else:
            feats = [0.0]*len(labels)
            
        feats_array = np.array([feats]) 
        probs = predict(models, feats_array)[0]
        results.append([fname] + list(probs))

    with open(args.output, 'w') as f:
        f.write("fasta_file\t"+"\t".join(labels)+"\n")
        for r in results:
            f.write(f"{r[0]}\t"+"\t".join([f"{x:.6f}" for x in r[1:]])+"\n")


if __name__ == "__main__":
    main()
