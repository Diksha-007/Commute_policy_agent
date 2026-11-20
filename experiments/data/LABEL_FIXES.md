# Label audit log

The long policies, queries and fact ledgers were written by another model. Before any experiment used
them, `experiments/validate_labels.py` checked them against the documents. This file lists every change
made after that audit, so the data can be traced back to what was generated.

Layout: `experiments/data/<company>_<version>_long.md | _queries.jsonl | _facts.json`.
Parked as unused (incomplete: its queries file was empty, and Acme v2 is the active version):
`_unused/acme_v1_*`.

## Changes to the generated data

| File | Change | Why |
|---|---|---|
| `acme_v2_queries.jsonl` (16 rows) | `must_contain_each` entries wrapped into groups, e.g. `"₹2,000"` -> `["₹2,000"]` | The model wrote bare strings; the format is a list of lists of alternative phrasings. Several rows had passed validation only by accident |
| `acme_v2_queries.jsonl` (acme-b4, 4 rows) | `required_sections` set to `[["7.2"]]` | The full first-to-fourth no-show ladder is complete only in the §7.2 table; §7.1 covers just the first two outcomes |
| `initech_v1_queries.jsonl` (initech-b4, 4 rows) | Golden text corrected to "The employee's shift manager must verify..."; `sections` reduced to `["5.4"]`; the §5.3 and §25.6 golden entries removed | The golden text was not verbatim, and neither §5.3 nor §25.6 states the "separate confirmation / does not carry over" fact |
| `initech_v1_facts.json` (initech-f004) | `section_ids` reduced to `["5.4"]`, `statement_count` 1 | Same reason: only §5.4 contains the fact |
| `validate_labels.py` seeds | Accept "90 calendar days" (Acme retention) | The document words it that way |

## Known deviations, left as they are

* Initech's weather trigger is "red weather alert" only (the seed table asked for "red alert plus police advisory"),
  so it matches Acme's trigger. No query depends on it; the policy text was not edited.
* Some `must_not_contain` phrases appear nowhere in their own company's policy (they are other companies'
  values or invented wrong answers). They still detect a wrong answer, but they are not look-alike traps.
* Globex's complaint-acknowledgement string "4 hours" also matches inside "24 hours" with naive substring
  search (sections 11.2, 25.2). All matching in the experiments is boundary-aware (see `contains_phrase`).
