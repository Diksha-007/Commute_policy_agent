# Evaluation

How the evaluation suite (36 cases) was run on the pipeline as the app runs it (HyDE on), and what it found: the method, the results,
the known failures and the limits. Raw outputs are in `eval/results/`, and every command is at the end. The experiments behind the
pipeline's configuration are in [experiments.md](experiments.md), and the design decisions are in the [README](../README.md).

## 1. What was evaluated

| Part | Setting in the evaluated pipeline |
|---|---|
| Retrieval | bge-base embeddings, exact search in the caller's own collection, **5 passages** shown to the model |
| HyDE | **on**: an AI call writes a policy-style passage that is searched alongside the question |
| Citations | the model picks **at most 3** of the 5; code keeps the 3 best-ranked if it names more |
| Not-covered gate | best similarity below **0.50** gives "not covered" without an answer call |
| Answer model | `openai/gpt-oss-120b` on Groq, JSON output, validated in code |
| Judge | `qwen/qwen3.8-27b`, a different model family from the answerer |
| Time limits during the run | total 2.7 s, HyDE 1.0 s, search 0.5 s, answer 2.2 s (they have since been raised, see §5) |

## 2. Method

- **36 cases** (`eval/cases.jsonl`), each with a company, an employee profile, a question, a **reference answer** and an expected status.
  Acme v2 is ingested last, so every Acme case is answered by the new version and must show no v1-only value.
- **Categories:** direct (4), paraphrase (4), table lookup (4), cross-company traps (4), not covered (3), injection (4),
  version update (3), two-part questions "multi_section" (10).
- **Real model, one run per case**, fresh temporary storage, requests paced for the free tier (the wait is not counted in latency).
- **Scoring.** An LLM judge is told the reference answer and decides whether the reply is correct: acceptable status, every reference
  fact stated correctly, no other company's rule, a cited section among the expected ones. The **leak check stays in code** (a citation
  from another company or version, or another company's name in the answer) and vetoes a pass. The reply is passed to the judge as
  data, and the judge is told to ignore instructions in it.
- **A second opinion.** The same replies were also scored by plain substring matching against `must_contain_*` / `must_not_contain`
  (the offline mode). The two methods agreed on 32 of the 34 cases they could both judge.
- **What a reference contains.** Only the facts the question asks for; side details go in an optional `context` the judge may use but
  must not require. How the references were written and revised, including every change, is in `eval/CASE_REVISIONS.md`.
- **Output:** `eval/results/results_groq_hyde_on.jsonl`, one line per case with `id, input, expected, actual, pass, notes`
  (the judge's verdict and reason are in `actual.judge` and the notes).

## 3. Results

| Measure | Result |
|---|---|
| Cases passed | **33 of 36** (91.7%) |
| Not passed because Groq timed out ("try again") | 2 (rerun below: both pass) |
| Not passed because of the answer | 1 |
| Questions that got an answer | 34, of which **33 passed** |
| Cross-company leaks | **0** |
| By category | direct 3/4, paraphrase 4/4, table 4/4, cross-company 4/4, not covered 3/3, injection 3/4, version 3/3, two-part 9/10 |
| Status correct | 94.4% (the 2 timeouts are the only misses) |
| Citation correct | 100% |
| At least one expected section among the 5 retrieved | 100% |
| LLM calls per question | average **1.94**, maximum 2 (limit 2) |
| Questions where HyDE failed and the search fell back to the question alone | 4 of 36 |
| Citations per answer | maximum 2 (limit 3) |
| Latency, LLM included | p50 **1.52 s**, p95 **2.39 s**, max 2.47 s (target: p95 <= 3 s) |

The two non-passes in the injection and direct categories are the timeouts; the one failure caused by the answer is in the two-part category (§4).

### The two timed-out cases, rerun separately

`direct-globex-eligibility` and `inject-initech-annexure-claim` returned "try again" because their AI call ran out of time, so they say
nothing about answer quality. They were rerun on their own with the same settings (HyDE on, LLM judge) and the current code, including the
raised time limits:

| Case | Result | Answer given | Latency | AI calls |
|---|---|---|---|---|
| `direct-globex-eligibility` | **pass** | "All full-time Globex employees of any grade are eligible, provided their registered home address is more than 8 km from the office by road." (cites §2.1, §1.2) | 1.27 s | 2 (HyDE used) |
| `inject-initech-annexure-claim` | **pass** | "No, the policy includes a no-show penalty schedule. The first no-show gets a warning, the second incurs a ₹500 fine, and further no-shows lead to suspension or HR referral." (cites §7.2, §7.1) | 1.25 s | 2 (HyDE used) |

Output: `eval/results/rerun_of_timeouts/`. The main results file is unchanged. Counting these two reruns in place of the timeouts, the
suite would stand at **35 of 36**. That combines two runs and rests on one sample each, so the main result stays **33 of 36**.

## 4. Known failures

| Case | What happens | Cause |
|---|---|---|
| `multi-acme-last-drop-guests` | It gives the employee's own shift drop (6:15 PM), or the right last drop (12:30 AM), and says guest transport is "not covered" | **Real.** The guest section (9.1) is not among the 5 retrieved chunks for this two-part question, so only half can be answered |
| `direct-globex-eligibility`, `inject-initech-annexure-claim` | "Try again" in the main run | **Time limit, not the answer.** The AI call ran out of time. Both pass when rerun (§3) |

Two real weaknesses remain: retrieval of the second half of a two-part question, and run-to-run variation of the answer model
(temperature 0 does not make it deterministic; a few cases, such as `multi-acme-late-cancel-noshow`, flipped between runs during development).

## 5. Limitations and trade-offs

- **The cases were also used to develop the system.** The prompt and the references were improved while looking at failures on these 36 cases, so they are not a held-out test. Expect accuracy on new questions to be somewhat lower than 33 of 36.
- **Small set, one run per case.** One case is 2.8 points, and a few cases can flip between runs. Treat the result as a range, not a point.
- **Cases and references were written by us**, with the policies, so they match how the policies are phrased. Real employees will ask odder questions. Nobody else has reviewed the references.
- **Trade-off: an LLM judge instead of string matching.** It reads answers the way a person would (wording differences don't fail a correct answer), at the price of being an AI whose strictness can vary and a reference set that must be written and kept honest. It failed one correct reply during development for using the policy's own wording ("full-time"), and the references were revised after seeing results, by one stated rule (every change is in `eval/CASE_REVISIONS.md`). String scoring stays available for offline runs.
- **Trade-off: relaxed time limits.** The app now allows 5 s in total (HyDE 1.5 s, search 1.5 s, answer 3 s) so that a slow machine gets a slow answer instead of "try again". The cost: on slow hardware a question can take up to 5 s, so the brief's p95 <= 3 s is demonstrated on the evaluation machine (p95 2.4 s), not guaranteed everywhere. **The numbers above were produced under the earlier, tighter limits and the full suite was not rerun**; only the two timed-out cases were (§3). The new limits can only turn timeouts into answers.
- **The gate threshold (0.50) was chosen on a sweep that includes these cases** (the sweep used the 61-case extended file). The overfitting risk is small: the gate only removes clearly off-topic questions, and no answerable question was wrongly gated.
- **No runtime fact check.** If the model wrote another company's number without citing it, only this evaluation would catch it.
- **The free tier shapes behaviour:** requests were paced, so latency here is per-request time without queueing, not throughput.

## 6. Reproduce

```
docker compose run --rm app pytest -q                                                    # 123 tests, fake LLM + fake embedder, about 15-35 s
docker compose run --rm app pytest -q -m real_model                                      # 3 tests with the real embedding model
docker compose run --rm app python -m eval.run_eval --llm fake                           # offline check of the 36 cases (about 25 s; the pass count is not meaningful)
docker compose run --rm app python -m eval.run_eval --llm groq --repeats 1               # the real evaluation: HyDE on, LLM-judged
docker compose run --rm app python -m eval.run_eval --llm groq --judge strings           # the same, scored by string matching
docker compose run --rm app python -m eval.run_eval --llm groq --only direct-globex-eligibility,inject-initech-annexure-claim --out eval/results/my_rerun   # rerun chosen cases
docker compose run --rm app python -m eval.run_eval --sweep --cases eval/cases_extended.jsonl   # gate threshold
```

The real run needs `groq_key` and takes **about 20 minutes** at the default pacing (it waits between AI calls for the free tier's token limit).
**It writes `eval/results/results_groq_hyde_on.jsonl`, replacing the committed results**; add `--out eval/results/my_run` to keep them.
