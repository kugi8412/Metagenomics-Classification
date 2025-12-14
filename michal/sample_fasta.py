#!/usr/bin/env python3
import sys
import random
import gzip

def open_maybe_gz(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def read_records(handle):
    """Generator FASTA: zwraca (header, seq)."""
    header = None
    seq_lines = []
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_lines)
            header = line
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        yield header, "".join(seq_lines)

def reservoir_sample_fasta(input_path, output_path, n_samples, seed=None):
    if seed is not None:
        random.seed(seed)

    reservoir = []
    total = 0

    with open_maybe_gz(input_path, "rt") as inp:
        for rec in read_records(inp):
            total += 1
            if len(reservoir) < n_samples:
                reservoir.append(rec)
            else:
                j = random.randint(0, total - 1)
                if j < n_samples:
                    reservoir[j] = rec

    out_fh = gzip.open(output_path, "wt") if output_path.endswith(".gz") else open(output_path, "w")
    with out_fh as out:
        for header, seq in reservoir:
            out.write(f"{header}\n")
            out.write(f"{seq}\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: sample_fasta.py input.fasta(.gz) output.fasta(.gz) N seed", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    N = int(sys.argv[3])
    seed = int(sys.argv[4])
    reservoir_sample_fasta(input_path, output_path, N, seed)
