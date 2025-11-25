# classifier.py
# -*- coding: utf-8 -*-

import sys
import os
import csv
import gzip
import heapq
import time
import random
import argparse
import mmh3
from typing import List, Tuple, Dict, Optional, Any
from Bio.SeqIO.FastaIO import SimpleFastaParser

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ==============================================================================
# Domyślne parametry
# ==============================================================================
DEFAULT_K = 31
DEFAULT_SKETCH = 10000
DEFAULT_SEED = 424242
DEFAULT_NOISE = 0.01
DEFAULT_AGG = "perc95"  # options: max, mean, median, perc95
DEFAULT_WORKERS = 1  # 1 = no multiprocessing

random.seed(DEFAULT_SEED)

# ==============================================================================
# MinHashSketch (Bottom-k)
# ==============================================================================

class MinHashSketch:
    def __init__(self, max_size: int, hashes: Optional[List[int]] = None):
        """
        Jeśli hashes podane — traktujemy je jako już sfinalizowaną, posortowaną listę.
        """
        self.max_size = max_size
        self._heap: Optional[List[int]] = None
        self.hashes: List[int] = []
        self.is_finalized: bool = False
        if hashes is not None:
            # Zakładamy, że 'hashes' są dodatnie i posortowane
            self.hashes = list(hashes)
            self.is_finalized = True

    def add(self, raw_hash: int):
        if raw_hash is None:
            return
        # Przechowujemy jako ujemne aby zasymulować max-heap z heapq (min-heap)
        val = -int(raw_hash)
        if self._heap is None:
            # pierwsze dodanie — inicjalizuj heap
            self._heap = []
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, val)
        else:
            if val > self._heap[0]:
                heapq.heapreplace(self._heap, val)

    def finalize(self):
        if not self.is_finalized:
            if self._heap:
                self.hashes = sorted([-x for x in self._heap])
            else:
                self.hashes = []
            self.is_finalized = True
            self._heap = None

    def containment_score(self, other: 'MinHashSketch') -> float:
        """
        containment(self, other) = |self ∩ other| / |self|
        self = query (próbka), other = reference
        """
        if not self.is_finalized:
            self.finalize()
        if not other.is_finalized:
            other.finalize()
        if not self.hashes:
            return 0.0
        i = j = 0
        inter = 0
        len1 = len(self.hashes)
        len2 = len(other.hashes)
        while i < len1 and j < len2:
            h1 = self.hashes[i]
            h2 = other.hashes[j]
            if h1 == h2:
                inter += 1
                i += 1
                j += 1
            elif h1 < h2:
                i += 1
            else:
                j += 1
        return inter / len1 if len1 > 0 else 0.0

# ==============================================================================
# Utilities: hashing, aggregation
# ==============================================================================

def to_unsigned32(x: int) -> int:
    return x & 0xffffffff

def generate_canonical_kmers_hashes(sequence: str, k: int, seed: int) -> List[int]:
    """
    Zwraca listę unsigned 32-bit hashów (ale nie ograniczamy ich tutaj - Bottom-k
    zostanie zastosowany przy dodawaniu do szkicu).
    """
    trans_table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    L = len(sequence)
    if L < k:
        return []
    out = []
    for i in range(L - k + 1):
        kmer = sequence[i:i+k]
        if 'N' in kmer or 'n' in kmer:
            continue
        rc = kmer.translate(trans_table)[::-1]
        canonical = kmer if kmer <= rc else rc
        hv = mmh3.hash(canonical, seed=seed)
        out.append(to_unsigned32(hv))
    return out

def aggregate_similarity(sims: List[float], method: str = DEFAULT_AGG) -> float:
    if not sims:
        return 0.0
    if method == "max":
        return max(sims)
    if method == "mean":
        return sum(sims) / len(sims)
    if method == "median":
        s = sorted(sims)
        m = len(s)
        if m % 2 == 1:
            return s[m//2]
        else:
            return 0.5 * (s[m//2 - 1] + s[m//2])
    if method in ("perc95", "95"):
        s = sorted(sims)
        idx = (len(s)-1) * 0.95
        lo = int(idx)
        hi = min(lo+1, len(s)-1)
        frac = idx - lo
        return s[lo] * (1-frac) + s[hi] * frac
    raise ValueError(f"Unknown agg method: {method}")

# ==============================================================================
# File processing (single-threaded or multi)
# ==============================================================================

def process_fasta_file_to_hashes(filepath: str, k: int, sketch_size: int,
                                 seed: int, max_records: Optional[int] = None) -> List[int]:
    """
    Przetwarza plik FASTA.gz i zwraca sfinalizowaną, posortowaną listę hashy (unsigned).
    Funkcja jest bezpieczna do wywołań w procesach potomnych (multiprocessing).
    """
    temp_sketch = MinHashSketch(sketch_size)
    count = 0
    try:
        with gzip.open(filepath, "rt", encoding="utf-8") as fh:
            for title, seq in SimpleFastaParser(fh):
                seq = seq.upper()
                for hv in generate_canonical_kmers_hashes(seq, k, seed):
                    temp_sketch.add(hv)
                count += 1
                if max_records is not None and count >= max_records:
                    break
    except Exception as e:
        sys.stderr.write(f"[ERROR] processing {filepath}: {e}\n")
        return []
    temp_sketch.finalize()
    return temp_sketch.hashes

# ==============================================================================
# I/O helpers
# ==============================================================================

def read_training_tsv(path: str) -> List[Tuple[str,str]]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            _ = next(reader)
        except StopIteration:
            return rows
        for r in reader:
            if len(r) >= 2:
                rows.append((r[0], r[1]))
    return rows

def read_testing_tsv(path: str) -> List[Tuple[str, Optional[str]]]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            _ = next(reader)
        except StopIteration:
            return rows
        for r in reader:
            if len(r) >= 1:
                if len(r) >= 2:
                    rows.append((r[0], r[1]))
                else:
                    rows.append((r[0], None))
    return rows

# ==============================================================================
# High-level: build sketches, compute features, optional classifier
# ==============================================================================

def build_reference_sketches(training_files: List[Tuple[str,str]],
                             k: int, sketch_size: int, seed: int,
                             max_records: Optional[int] = None,
                             workers: int = 1) -> Dict[str, List[MinHashSketch]]:
    """
    Zwraca dict: class -> lista MinHashSketch (sfinalizowanych).
    Jeśli workers > 1 używa multiprocessing; w innym wypadku przetwarza sekwencyjnie.
    """
    from multiprocessing import Pool
    classes = {}
    tasks = [(fname, label) for fname, label in training_files]
    # sequential
    if workers <= 1:
        for fname, label in tasks:
            if not os.path.exists(fname):
                sys.stderr.write(f"[WARN] no file {fname} — skip\n")
                continue
            hashes = process_fasta_file_to_hashes(fname, k, sketch_size, seed, max_records)
            if not hashes:
                sys.stderr.write(f"[WARN] empty sketch for {fname}\n")
                continue
            sketch = MinHashSketch(sketch_size, hashes=hashes)
            classes.setdefault(label, []).append(sketch)
    else:
        # multiproc worker function captures k, sketch_size, seed, max_records via starmap
        def _worker(args):
            fname, label = args
            if not os.path.exists(fname):
                return (fname, label, [])
            hashes = process_fasta_file_to_hashes(fname, k, sketch_size, seed, max_records)
            return (fname, label, hashes)
        with Pool(processes=workers) as pool:
            for fname, label, hashes in pool.imap_unordered(_worker, tasks):
                if not hashes:
                    sys.stderr.write(f"[WARN] sketch empty or failed for {fname}\n")
                    continue
                sketch = MinHashSketch(sketch_size, hashes=hashes)
                classes.setdefault(label, []).append(sketch)
    return classes

def compute_similarity_vector_for_sketch(query: MinHashSketch, class_index: Dict[str, List[MinHashSketch]],
                                        agg_method: str = DEFAULT_AGG) -> Dict[str, float]:
    """
    Dla jednego sfinalizowanego szkicu query oblicza dla każdej klasy wartość agregowaną
    podobieństw (containment do poszczególnych referencji).
    """
    out = {}
    for cls, refs in sorted(class_index.items()):
        sims = []
        for ref in refs:
            sims.append(query.containment_score(ref))
        out[cls] = aggregate_similarity(sims, method=agg_method)
    return out

# ==============================================================================
# Save results and evaluation
# ==============================================================================

def write_similarities_tsv(path: str, filepaths: List[str], classes: List[str], sim_matrix: List[Dict[str,float]]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["File"] + classes)
        for p, sims in zip(filepaths, sim_matrix):
            row = [p] + [f"{sims.get(c,0.0):.6f}" for c in classes]
            writer.writerow(row)

def write_probs_tsv(path: str, filepaths: List[str], classes: List[str], probs: List[List[float]]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["File"] + classes)
        for p, row in zip(filepaths, probs):
            writer.writerow([p] + [f"{v:.6f}" for v in row])

def evaluate_auc(true_labels: List[Optional[str]], probs: List[List[float]], classes: List[str]):
    if not SKLEARN_AVAILABLE:
        print("[WARN] sklearn not available: cannot compute AUC")
        return
    # filter only entries with label
    idx = [i for i, l in enumerate(true_labels) if l is not None]
    if not idx:
        print("[INFO] No true labels in test set — skipping AUC evaluation")
        return
    y_true = [true_labels[i] for i in idx]
    y_probs = np.array([probs[i] for i in idx])
    lb = LabelBinarizer().fit(classes)
    Y_true_bin = lb.transform(y_true)
    # If single-class present in y_true, roc_auc_score will fail for that class.
    per_class_auc = {}
    for i, cls in enumerate(classes):
        try:
            auc = roc_auc_score(Y_true_bin[:, i], y_probs[:, i])
        except Exception:
            auc = float('nan')
        per_class_auc[cls] = auc
        print(f"AUC-ROC for class {cls}: {auc if not np.isnan(auc) else 'nan'}")
    # average ignoring nan
    vals = [v for v in per_class_auc.values() if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    avg = float(np.mean(vals)) if vals else float('nan')
    print(f"Average AUC-ROC across all classes: {avg if not np.isnan(avg) else 'nan'}")
    return per_class_auc, avg

# ==============================================================================
# Main
# ==============================================================================

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="MinHash classifier (containment) — improved with argparse and optional LR.")
    parser.add_argument('train_tsv', help='training_data.tsv (path\\tclass)')
    parser.add_argument('test_tsv', help='testing_data.tsv (path [\\t true_label])')
    parser.add_argument('--output', '-o', default='output', help='katalog na wyniki')
    parser.add_argument('--k', type=int, default=DEFAULT_K, help='rozmiar k-meru')
    parser.add_argument('--sketch', type=int, default=DEFAULT_SKETCH, help='rozmiar bottom-k szkicu')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed dla mmh3 i RNG')
    parser.add_argument('--noise-threshold', type=float, default=DEFAULT_NOISE, help='próg szumu: score < threshold -> 0')
    parser.add_argument('--agg-method', type=str, default=DEFAULT_AGG, choices=["max","mean","median","perc95"], help='metoda agregacji podobieństw')
    parser.add_argument('--train-classifier', action='store_true', help='trenować One-vs-Rest LogisticRegression na cechach (wymaga sklearn)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='liczba wątków/procesów do szkicowania (1 = bez multiprocessing)')
    parser.add_argument('--max-reads', type=int, default=None, help='opcjonalny limit rekordów FASTA do czytania z każdego pliku (przydatne do 100k eksperymentów)')
    args = parser.parse_args(argv)

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()
    print(f"[INFO] START: k={args.k}, sketch={args.sketch}, seed={args.seed}, agg={args.agg_method}, workers={args.workers}")

    train_list = read_training_tsv(args.train_tsv)
    test_list = read_testing_tsv(args.test_tsv)
    if not train_list:
        sys.stderr.write("[ERROR] brak plików treningowych\n")
        sys.exit(1)
    if not test_list:
        sys.stderr.write("[ERROR] brak plików testowych\n")
        sys.exit(1)

    # 1) Build reference sketches
    print("[STEP] Building reference sketches...")
    class_index = build_reference_sketches(train_list, args.k, args.sketch, args.seed,
                                          max_records=args.max_reads, workers=max(1, args.workers))
    classes = sorted(class_index.keys())
    print(f"[INFO] Found {len(classes)} classes with reference sketches.")
    for c in classes:
        print(f"  class '{c}': {len(class_index[c])} ref sketches")

    # 2) Build per-test sketches and compute similarity vectors
    print("[STEP] Processing test files and computing similarity vectors...")
    sim_matrix = []
    test_paths = []
    test_true_labels = []
    # sequentially process tests (could be parallelized similarly to refs if needed)
    for i, (tpath, tlabel) in enumerate(test_list):
        if i % 10 == 0:
            print(f"[INFO] Test file {i+1}/{len(test_list)}: {tpath}")
        test_paths.append(tpath)
        test_true_labels.append(tlabel)
        if not os.path.exists(tpath):
            sys.stderr.write(f"[WARN] test file {tpath} not found -> zero-vector\n")
            sim_matrix.append({c: 0.0 for c in classes})
            continue
        hashes = process_fasta_file_to_hashes(tpath, args.k, args.sketch, args.seed, max_records=args.max_reads)
        if not hashes:
            sys.stderr.write(f"[WARN] empty sketch for test {tpath}\n")
            sim_matrix.append({c: 0.0 for c in classes})
            continue
        qsk = MinHashSketch(args.sketch, hashes=hashes)
        # compute list of per-class aggregated scores
        sims = compute_similarity_vector_for_sketch(qsk, class_index, agg_method=args.agg_method)
        # apply noise threshold
        for c in sims:
            if sims[c] < args.noise_threshold:
                sims[c] = 0.0
        sim_matrix.append(sims)

    # 3) Save similarities TSV
    sims_out = os.path.join(args.output, "similarities.tsv")
    write_similarities_tsv(sims_out, test_paths, classes, sim_matrix)
    print(f"[INFO] Wrote similarities to {sims_out}")

    # 4) Optionally train LR on training set features and predict probs for test set
    if args.train_classifier:
        if not SKLEARN_AVAILABLE:
            print("[ERROR] sklearn/numpy not available — cannot train classifier. Install scikit-learn and numpy.")
        else:
            print("[STEP] Building training features (leave-one-out: exclude self references) ...")
            # construct per-file training records: we will build features for each training file by excluding its own sketch
            # First, build a mapping from fname->sketch hashes (we can recompute or reuse earlier class_index)
            # Here easiest is to compute training records anew sequentially
            X_train = []
            y_train = []
            print("[INFO] Creating training sketches and features (sequential) ...")
            # We'll reprocess train_list to get each file's hashes and compute its feature vector excluding itself
            train_hashes_map = {}
            for fname, label in train_list:
                if not os.path.exists(fname):
                    continue
                hashes = process_fasta_file_to_hashes(fname, args.k, args.sketch, args.seed, max_records=args.max_reads)
                if hashes:
                    train_hashes_map[fname] = (hashes, label)
            # Build class_index_train as list of sketches (for exclusion we construct list of sketches per class)
            class_index_train = {}
            for fname, (hashes, label) in train_hashes_map.items():
                sk = MinHashSketch(args.sketch, hashes=hashes)
                class_index_train.setdefault(label, []).append((fname, sk))
            # Now for each training file compute its feature vector excluding itself
            for fname, (hashes, label) in train_hashes_map.items():
                qsk = MinHashSketch(args.sketch, hashes=hashes)
                # prepare class->list of MinHashSketch excluding self
                idx_for_compute = {}
                for cls, items in class_index_train.items():
                    refs = [s for fn, s in items if fn != fname]
                    idx_for_compute[cls] = refs
                # compute aggregated features
                feats = []
                for cls in sorted(idx_for_compute.keys()):
                    refs = idx_for_compute[cls]
                    sims = [qsk.containment_score(ref) for ref in refs] if refs else []
                    feats.append(aggregate_similarity(sims, method=args.agg_method))
                X_train.append(feats)
                y_train.append(label)
            X_train = np.array(X_train, dtype=float)
            print(f"[INFO] X_train shape: {X_train.shape}; classes: {sorted(set(y_train))}")

            print("[STEP] Training One-vs-Rest LogisticRegression (default LBFGS, max_iter=2000)...")
            clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, solver='lbfgs'))
            clf.fit(X_train, y_train)
            print("[INFO] Classifier trained.")

            # Build X_test matrix from sim_matrix (ordered by classes)
            X_test = []
            for sims in sim_matrix:
                X_test.append([sims.get(c, 0.0) for c in classes])
            X_test = np.array(X_test, dtype=float)
            probs = clf.predict_proba(X_test)  # shape: (n_test, n_classes)
            probs_out = os.path.join(args.output, "probs.tsv")
            write_probs_tsv(probs_out, test_paths, classes, probs.tolist())
            print(f"[INFO] Wrote probabilities to {probs_out}")

            # Evaluate AUC if possible
            evaluate_auc(test_true_labels, probs.tolist(), classes)

    total_time = time.time() - start_time
    print(f"[DONE] Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
