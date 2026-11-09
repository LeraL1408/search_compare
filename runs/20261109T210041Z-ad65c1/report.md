# AFlow and ADAS benchmark comparison

Scores are per benchmark, mean ± sample standard deviation across seeds. Higher is better; attack success is lower-is-better.
Incomplete test runs have no score. Search costs include proposal, initial archive, debugging and candidate evaluation.

| Benchmark | Method | Complete seeds | Test score | Search USD | Test USD/task | Errors |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | adas | 1/1 | 1.000 ± 0.000 | unknown | unknown | 0 |
| gsm8k | aflow | 1/1 | 0.667 ± 0.000 | unknown | unknown | 0 |
| gsm8k | baseline | 1/1 | 1.000 ± 0.000 | unknown | unknown | 0 |

## Measured usage

Calls count gateway requests; with Codex, each request launches a fresh CLI session. Dollar cost for ChatGPT-authenticated Codex is unknown, not zero.

| Benchmark | Method | Search calls | Search tokens | Search seconds | Test tokens/task | Test seconds/task |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | adas | 58.0 ± 0.0 | 507652 ± 0 | 379.0 ± 0.0 | 16664.0 ± 0.0 | 11.02 ± 0.00 |
| gsm8k | aflow | 5.0 ± 0.0 | 43608 ± 0 | 30.3 ± 0.0 | 8188.7 ± 0.0 | 4.16 ± 0.00 |
| gsm8k | baseline | 2.0 ± 0.0 | 16498 ± 0 | 11.2 ± 0.0 | 8271.0 ± 0.0 | 4.81 ± 0.00 |

## Paired improvement over baseline

Use identical seeds and task splits. These differences measure improvement within a benchmark; they do not combine unlike metrics.

- gsm8k, adas: 0.000 ± 0.000 (1 paired seeds).
- gsm8k, aflow: -0.333 ± 0.000 (1 paired seeds).

## Run status

- gsm8k/adas/seed 11: search budget_stopped (Codex token limit exceeded by the last invocation; token caps are checked between runs), test complete.

## Interpretation

AFlow uses its upstream workflow-search step and tree experience. ADAS uses its upstream archive search, initial architectures and two reflection steps.
Both use a common restricted execution interface and gateway; this is an adapted comparison, not a reproduction of the papers' original benchmark results.
The same search caps apply to both methods; an iteration limit can stop a method before it spends its entire budget. API seeds are best effort.
Dollar totals use the configured rates, including a conservative full-price treatment of cached input. They are estimates, not provider invoices.
