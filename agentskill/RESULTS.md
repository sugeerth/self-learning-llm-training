# agentskill — closed-loop evaluation report

Seeds: [0, 1, 2, 3, 4] · transient tool fail rate: 15% · every number is mean ± 95% CI over seeds.

**Mined recovery rule** (from trajectories, not hardcoded): retry=True, P(retry succeeds)=100%, support=40 events.

## Ablations — which learned skill contributes what

| policy | success rate | tool efficiency | avg steps |
|---|---|---|---|
| baseline | 24% ± 10% | 96% ± 0% | 4.7 |
| +plans | 45% ± 8% | 99% ± 1% | 5.2 |
| +recovery | 50% ± 0% | 96% ± 0% | 5.4 |
| learned (full) | 100% ± 0% | 99% ± 1% | 6.0 |

`+plans` isolates planning/tool-selection learned from trajectories; `+recovery` isolates the mined retry skill; the full agent composes both.

## Transfer — leave-one-domain-out

Memory and recovery are learned WITHOUT the held-out domain, then evaluated on it. Domain plans should vanish; the procedural recovery skill should survive.

| held-out benchmark | baseline | learned (LODO) | lift |
|---|---|---|---|
| GAIA | 26% | 50% | +24% |
| MLE-bench | 26% | 50% | +24% |
| SWE-bench Verified | 20% | 50% | +30% |

## Mined sub-skills (top tool bigrams per domain)

- **MLE-bench**: load_data→eda (30), eda→train (30), evaluate→submit (30)
- **SWE-bench Verified**: locate→read_code (28), edit→run_tests (28), run_tests→commit (28)
- **GAIA**: search→read (26), read→extract (14), extract→answer (14)

*Synthetic trajectories + mock suites (no GPU/network/key in this sandbox); magnitudes are properties of the mock — the mechanism, ablation structure, and transfer split are the result. Real data/benchmarks plug in via sources.py and benchmark.py.*
