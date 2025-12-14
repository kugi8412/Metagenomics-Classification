import os
import subprocess

train = "./train0_data.tsv"
test = "./extra_test_data.tsv"

k_values = [17, 18, 19, 20, 21]
sketch_sizes = [8000, 9000, 10000, 11000, 12000]

for k in k_values:
    for sketch in sketch_sizes:
        output = f"results_k{k}_s{sketch}.tsv"

        print(f"\n=== Running k={k}, sketch={sketch} ===\n")

        cmd = [
            "python3",
            "classifier.py",
            train,
            test,
            output,
            "--k", str(k),
            "--sketch_size", str(sketch)
        ]

        subprocess.run(cmd)
