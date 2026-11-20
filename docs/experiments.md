# Experiments

This document records the five experiments (E1 to E5) carried out before the design of the assistant was fixed. For each experiment it states
the objective, the methodology and the results. The decisions taken on the basis of these experiments are described in the
[README](../README.md). The scripts and their inputs are in `experiments/`; their outputs are written to `experiments/results/` when they are
run and are not committed.

## Common setup (E1 to E4)

- **Scope.** The experiments measure which passages reach the LLM (retrieval), not the quality of the final answers.
- **Documents.** Three synthetic company policies of approximately 10,000 words each (Acme v2, Globex, Initech), divided by the application's
  own chunker into about 530 passages (`experiments/data/*_long.md`).
- **Questions.** 48 base questions, each asked in its original wording and in a keyword, a chatty and a synonym rewording. E3 and E4 used four
  times as many questions (192). Nine questions not covered by the policies were used only for the "not covered" check in E1.
- **Relevance.** A passage counted as containing a fact when it belonged to a section labelled for that fact and contained the fact's key text,
  matched on word boundaries (so that "4 hours" does not match within "24 hours"). No LLM decided relevance.
- **Metrics.** Recall@k is the share of needed facts present in the top k passages. Differences were paired by question. The 95% intervals are
  bootstrap intervals that resample whole base questions, because the rewordings of one question are not independent.
- **Data quality.** The documents, questions and labels were written by an LLM and checked by script (`experiments/validate_labels.py`; corrections
  are logged in `experiments/data/LABEL_FIXES.md`). They were not reviewed independently. Timings were taken on a single Windows machine running Docker.

## E1: embedding model

**Objective.** To compare `all-MiniLM-L6-v2` (MiniLM) and `BAAI/bge-base-en-v1.5` (bge-base) as the embedding model.

**Methodology.** The 529 passages and the 48 answerable questions were embedded with each model (bge-base questions with its query instruction)
and searched exactly. The passages, labels and questions were identical for both models. Timings were measured with the two setups interleaved.

**Results.**

| Measure | MiniLM | bge-base |
|---|---|---|
| Recall@5 | 77% | 87% |
| Recall@3 | 70% | 76% |
| MRR | 0.69 | 0.71 |
| Reworded questions answered fully (top 5) | 61% | 75% |
| Synonym rewordings answered fully (top 5) | 42% | 67% |
| Original wording answered fully (top 5) | 100% | 100% |
| Time to embed one question | 4 ms | 22 ms |
| Model size | 90 MB | 210 MB |
| Separating covered from not-covered questions by similarity (AUC) | 0.91 | 0.79 |

bge-base was better on 7 questions, worse on 1 and equal on 40; the 95% interval of the difference touched zero. MiniLM run through Chroma's
wrapper took about 24 ms per question. With the 0.30 similarity cutoff, none of the 9 not-covered questions was refused on the long documents
(the lowest not-covered score was 0.337).

## E2: number of passages shown to the LLM

**Objective.** To compare showing the LLM three passages chosen for diversity (MMR) with showing it five, given that at most three may be cited.

**Methodology.** The data of E1 with bge-base, in four setups: **A**, the 3 most similar passages; **B**, 3 passages chosen by MMR from a pool of 10;
**C**, the 5 most similar; **D**, 5 chosen by MMR from a pool of 15. The MMR parameter (lambda) was chosen on Globex alone and checked on Acme and
Initech. A further sweep covered 32 combinations of lambda and pool size.

**Results.**

| | A: top 3 | B: 3 + MMR | C: top 5 | D: 5 + MMR |
|---|---|---|---|---|
| Facts found (average) | 0.76 | 0.76 | 0.87 | 0.82 |
| Questions with every needed fact retrieved | 67% | 69% | 81% | 77% |
| Passage text sent (approximate tokens) | 385 | 385 | 644 | 648 |
| Acme and Initech only (not used for tuning) | 0.78 | 0.83 | 0.89 | 0.88 |

Setup C found 0.10 more facts than setup A: it was better on 7 questions and worse on none, and the interval excluded zero. Two-part questions
rose from 0.54 to 0.71 and table questions from 0.67 to 0.92. None of the 32 lambda and pool combinations exceeded plain top 5.

## E3: HyDE with bge-base

**Objective.** In HyDE (Hypothetical Document Embeddings) an LLM writes a policy-style passage that answers the question, and the passage is searched
alongside the question. The objective was to determine whether this retrieves the needed passages more often.

**Methodology.** The 192 questions, bge-base and the four setups of E2, each searched three ways: question only, question and passage fused by
reciprocal rank fusion, and passage only. 201 passages were written once by `qwen/qwen3.8-27b` and cached in `experiments/data/hyde_passages.jsonl`.
The decision rule was fixed before the run: HyDE was to be adopted only if, for setup C, recall rose by at least 0.05 and the interval excluded zero.

**Results.**

| Facts found | A: top 3 | B: 3 + MMR | C: top 5 | D: 5 + MMR |
|---|---|---|---|---|
| Question only | 0.88 | 0.87 | 0.92 | 0.92 |
| Question and HyDE passage | 0.86 | 0.79 | 0.93 | 0.88 |
| HyDE passage only | 0.80 | 0.78 | 0.87 | 0.86 |

For setup C recall changed by +0.005 (interval −0.029 to +0.042; better on 7 questions, worse on 6), so the rule was not met. For setup B the
change was −0.08 with an interval excluding zero. A HyDE call took 0.22 s typically (95th percentile 0.68 s). Of the 201 passages, 25 (12%) contained
a number, time or amount that appears neither in the question nor in the policy. In an initial run on the original 48 questions, setup D showed
+0.04; on the 144 added questions it showed −0.06.

## E4: HyDE with MiniLM

**Objective.** To determine whether HyDE helps the weaker embedder (MiniLM) more than it helps bge-base.

**Methodology.** The data, passages and decision rule of E3, with MiniLM in place of bge-base; both embedders were run in one script.

**Results.**

| Setup C, facts found | Without HyDE | With HyDE | Change (95% interval) |
|---|---|---|---|
| MiniLM | 0.854 | 0.883 | +0.029 (+0.000 to +0.057); better on 12, worse on 4 |
| bge-base | 0.924 | 0.930 | +0.005 (−0.029 to +0.042) |

The rule was not met. For MiniLM, synonym rewordings rose from 0.760 to 0.833 and chatty rewordings from 0.823 to 0.885; the change was +0.094 on the
original 48 questions and +0.007 on the 144 added ones. The difference between the two embedders' HyDE effects was +0.023 (−0.021 to +0.065).
Without HyDE, bge-base exceeded MiniLM by 0.070 (+0.029 to +0.117); MiniLM with HyDE (0.883) differed from bge-base without it (0.924) by −0.042
(−0.091 to +0.003).

## E5: groundedness of answers

**Objective.** To determine whether each claim in the generated answers is supported by the retrieved passages, and whether this can be checked
automatically.

**Methodology.** The final pipeline answered 3 questions: a profile-dependent rule, a table lookup and a two-part question. An extractor (Qwen,
given only the answer) listed the separate claims. A verifier (Qwen) was given the 5 retrieved chunks and the employee profile, and labelled each
claim *supported*, *derived* (following from a stated rule), *unsupported* or *contradicted*, citing a chunk id and a quotation. Code verified that
each quotation appears in the named chunk (otherwise a "supported" label was downgraded) and recorded whether the chunk had been cited by the
answer. The judge is of a different model family from the answerer. To test the judge, each answer was also checked in two altered versions: one
number changed to a wrong value, and one invented policy-style sentence appended.

**Results.**

| Measure | Result |
|---|---|
| Claims in the 3 answers | 5, all grounded (3 supported, 2 derived) |
| False alarms; quotations not found in the chunk | 0; 0 |
| Altered answers: invented sentence flagged | 3 of 3 |
| Altered answers: wrong number flagged | 2 of 3 |
| Claims supported only by a retrieved chunk that the answer did not cite | 1 of 5 |

The wrong number that was not flagged changed "after 8:00 PM" to "after 10:00 PM"; the judge labelled the altered claim *derived*.

## Reproducing the experiments

The scripts run in the application's own image with the project mounted, so nothing further needs to be installed.

```
docker compose build
docker compose run --rm -v "$PWD:/app" app python /app/experiments/run_e1.py --models minilm bge-base   # about 10 min; downloads MiniLM on first use
docker compose run --rm -v "$PWD:/app" app python /app/experiments/run_e2.py                              # about 2 min; add --sweep for the lambda and pool sweep
docker compose run --rm -v "$PWD:/app" app python /app/experiments/run_e3.py --extended                   # about 10 min
docker compose run --rm -v "$PWD:/app" app python /app/experiments/run_e4.py                              # about 3 min
docker compose run --rm -v "$PWD:/app" app python /app/experiments/run_e5.py                              # needs groq_key; about 21 LLM calls, 4 min
```

E1 to E4 need no API key (the HyDE passages are shipped) and reproduced the figures above when rerun from a clean checkout. E5 depends on live model
replies, so its figures vary slightly between runs. The timing control for E1 is `experiments/latency_check.py`.

| Path | Content |
|---|---|
| `experiments/run_e1.py` to `run_e5.py` | one script per experiment |
| `experiments/offline.py`, `metrics.py`, `matching.py` | shared selection logic, metrics and string matching |
| `experiments/validate_labels.py`, `experiments/data/LABEL_FIXES.md` | label audit and its log |
| `experiments/data/*_long.md` | the three long policies |
| `experiments/data/*_queries*.jsonl`, `*_facts*.json` | the questions and the fact ledger they are labelled against |
| `experiments/data/hyde_passages.jsonl` | the cached HyDE passages |
| `experiments/data/make_extra_queries.py` | generator of the added questions |
