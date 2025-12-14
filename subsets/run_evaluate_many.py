import os
import subprocess
import sys

results_dir = sys.argv[1]
ground_truth = sys.argv[2]

files = sorted([
    f for f in os.listdir(results_dir)
    if f.startswith("results_k") and f.endswith(".tsv")
])

if not files:
    print("Brak plików results_k*.tsv w folderze:", results_dir)
    sys.exit(1)

print(f"Znaleziono {len(files)} plików wynikowych.\n")

for fname in files:
    path = os.path.join(results_dir, fname)

    print(f"=== Evaluating {fname} ===")

    subprocess.run([
        "python3",
        "evaluate.py",
        path,
        ground_truth
    ])

    print()
