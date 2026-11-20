"""Query-embedding latency, measured so that machine drift cannot favour one setup.

    python experiments/latency_check.py

The three setups are timed in alternating rounds (A, B, C, A, B, C, ...), not one after another, and each
round's median is shown, so a slow period hits all of them. E1 measured bge-base at 45 ms and E2 at 21 ms in
separate runs; this exists to say which is closer to the truth.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import metrics as M  # noqa: E402
from experiments.run_e1 import BGE_QUERY_INSTRUCTION, DATA, build_fastembed, build_minilm  # noqa: E402

ROUNDS = 5


def main() -> None:
    questions = [json.loads(line)["question"] for p in sorted(DATA.glob("*_queries.jsonl"))
                 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    specs = [
        build_minilm(),
        build_fastembed("minilm-fastembed", "sentence-transformers/all-MiniLM-L6-v2", 384, "", local_only=False),
        build_fastembed("bge-base", "BAAI/bge-base-en-v1.5", 768, BGE_QUERY_INSTRUCTION),
    ]
    names = {"minilm": "MiniLM via Chroma (the app today)", "minilm-fastembed": "MiniLM via fastembed",
             "bge-base": "bge-base via fastembed"}
    for spec in specs:                      # warm-up
        for text in questions[:5]:
            spec.query(text)
    samples = {s.key: [] for s in specs}
    per_round = {s.key: [] for s in specs}
    for _ in range(ROUNDS):
        for spec in specs:
            this = []
            for text in questions:
                start = time.perf_counter()
                spec.query(text)
                this.append((time.perf_counter() - start) * 1000)
            samples[spec.key] += this
            per_round[spec.key].append(M.percentile(this, 50))
    print(f"{len(questions)} questions x {ROUNDS} interleaved rounds per setup\n")
    print("| setup | p50 (ms) | p95 (ms) | mean (ms) | median of each round (ms) |")
    print("|---|---|---|---|---|")
    for spec in specs:
        s = samples[spec.key]
        rounds = ", ".join(f"{x:.1f}" for x in per_round[spec.key])
        print(f"| {names[spec.key]} | {M.percentile(s, 50):.1f} | {M.percentile(s, 95):.1f} | {np.mean(s):.1f} | {rounds} |")


if __name__ == "__main__":
    main()
