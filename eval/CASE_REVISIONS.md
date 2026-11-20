# How the reference answers in `cases.jsonl` were written and revised

Each case has a `reference` (the facts the question asks for, which the judge requires) and an optional `context` (true
extra detail the judge may use to recognise a correct answer but must not require). Cases are scored by an LLM judge that is
given these (`eval/judge.py`); the leak check stays in code. The string checks (`must_contain_*`) are used only by the offline
`--judge strings` mode.

**The rule:** `reference` contains only what the question asks. Side details (who also approves, who is notified, booking
channels, what happens at other tiers, the complement of a stated condition) belong in `context`.

## Revisions, in order (all disclosed; none loosened a security check)

1. **First version** (references written in full sentences, including side details). First judge run: **22/36**. Nine cases
   failed only because the reply left out a side detail that the question did not ask about, four because of infrastructure
   errors (three provider "try again", one unreadable judge reply), and one was a real miss.
2. **Rule applied to all 36 references at once** (side details moved to `context`; the judge prompt told that `context` is
   optional). Second run: 31/36. Of the five non-passes, two were references that still broke the rule
   (`table-globex-second-noshow`, `cross-globex-acme-saturday`): moved the extras to `context`.
3. **`direct-globex-eligibility`:** the judge failed it once for the word "full-time" (from the policy's own scope section) and
   once for the missing complement ("8 km or closer are not provided"). Moved the complement and the scope note to `context`.
4. **Two more references that still broke the rule** (found in the last run, after the answer prompt and timeout changes):
   `version-acme-second-noshow` (required "recovered through payroll", a side detail) and `inject-initech-doc-noshow`
   (required the first-tier warning, though the question asks for the penalty). Both moved to `context`. These were
   the last rubric changes; the judge's strictness still differs a little from run to run, so a case can flip between runs.
5. **Not changed on purpose:** `multi-acme-last-drop-guests` stays failing. The guest-travel section is not among the retrieved
   chunks, so the assistant says it is not covered: a real retrieval limit, not a rubric problem.

A note on reliability: with temperature 0 the answer model is still not fully deterministic. `multi-acme-late-cancel-noshow`
once reversed the 1-hour cancellation window (1 wrong in 5 samples) though it was right in the other four.

## Run history (all used the judge; counts are cases passed of 36; runs 1 to 7 had HyDE off, run 8 has it on)

| Run | State | Passed | Infrastructure errors | Real answer problems |
|---|---|---|---|---|
| 1 | first references, old prompt | 22 | 4 | 1 real, 9 reference faults |
| 2 | references trimmed, new prompt | 31 | 0 | 2 reference faults, 1 judge too literal, 2 answer problems |
| 3 | + 2 references fixed | 31 | 3 timeouts | 1 judge too literal, 1 answer problem |
| 4 | same | 30 | 4 timeouts | 2 answer problems |
| 5 | same, slower pacing | 29 | 5 timeouts | 2 answer problems |
| 6 | answer timeout 1.8 s -> 2.2 s | **33** | **0** | 3 answer problems |
| 7 | + prompt hint on threshold comparison and profile use | 30 | 3 timeouts | 1 answer problem, 2 reference faults (fixed since) |
| 8 | **HyDE on (the final pipeline)**, references as committed | **33** | 2 timeouts | 1 answer problem (`multi-acme-last-drop-guests`) |

The "answer problems" are the same few cases: `multi-acme-last-drop-guests` (guest section not retrieved, and the model sometimes
answers with the employee's shift drop), `multi-acme-late-cancel-noshow` (the cancellation window sometimes reversed, not seen
again after the prompt hint: 0 wrong in 7 answered samples) and `cross-globex-escort-acme-wording` (once an irrelevant profile detail,
once `not_covered`). Timeouts depend on Groq's free-tier speed at the time, not on the assistant.
