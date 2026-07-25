# Deep-Dive: What This System Actually Is, How It Works, and What's Broken

*A ground-truth investigation of `self-learning-llm-training`, traced from the code (not the docstrings) across all five subsystems, with file:line evidence. Every load-bearing claim below was verified against the actual source or by running the code.*

---

## 0. The one-paragraph truth

This repo presents as a **self-learning multi-agent LLM training loop**: Trainer → Evaluator → Judge → MetaJudge agents that drive, critique, and improve their own training, with a throughput harness, a Hyperband search that "beats random," a synthetic-data flywheel that resists model collapse, and a model-onramp router that learns which model to use. **The engineering scaffolding is real and mostly well-built, but the "self-learning" is far narrower than advertised, and several headline numbers are compromised by measurement bugs.** The only component that actually compounds knowledge across runs is the `CheapPrior` surrogate (`prior_store.json`). The agent hierarchy audits a value it didn't produce and feeds nothing back; the onramp's learning loop is structurally open in the wired path; the flywheel's "collapse fix" confounds two variables and never actually reaches collapse; and the search's "speedup vs random" is currently inflated by an apples-to-oranges vocabulary mismatch. Underneath, the pipeline silently runs **byte-level tokenization while claiming GPT-2 BPE**, which makes every perplexity number per-byte and uninterpretable as stated.

---

## 1. What it claims vs. what it is

| Claim (README/docstrings) | Reality (code) | Evidence |
|---|---|---|
| Trainer→Evaluator→Judge→MetaJudge **drives** training and gates it | The chain **audits Hyperband's winner**; it trains nothing it proposes and no verdict changes control flow | `agents.py:360-363`; runner lambda ignores cfg/lr `self_learning_runner.py:178-181`; dead adapter `:91-97` |
| Hyperband/prior **measurably beat random** (regret vs random) | The arms don't eval on the same vocab — random/evolve at 50304, hyperband/prior at 128 — so the speedups are a ppl-scale artifact | `harness.py:456,480` clamp; `arms.py` run_random no clamp; `evolve.py:94` hardcodes 50304; verified `effective_vocab=128`, ~55× ppl gap |
| GPT-2 BPE tokenizer (50257 vocab) | **Byte-level tokenizer** is active; bins contain ids 10–122 | `data.py:81-86` fallback; bins `min=10,max=122,~65 unique` |
| Synthetic-data flywheel **flattens model collapse** (+1.15%→+0.49%) | Both runs are sub-threshold **"stable" (no collapse)**, and the comparison confounds mode-change **with** a perplexity-gate toggle | `flywheel.py:315-317,396`; reports carry `verdict:"stable"`, one has `ppl_gate:true` the other absent |
| model-onramp: models **earn traffic** and autopilot promotes on evidence | The repo calls the SDK directly, bypassing the router's breaker/failover/**exploration**; candidates get 0 traffic, so autopilot never promotes; `.onramp/` doesn't even exist | `agents.py:137`; `onramp_bridge.py:60`; `routing.py:76-77`; `autopilot.py:57-59` |
| GRPO / reasoning / speculative decoding | All three are **standalone demo modules wired into nothing**; GRPO is un-clipped REINFORCE; spec-decode is not distribution-exact by its own comment | `grpo.py` imported nowhere executable; `speculative.py:120-124` |

**What this means:** the system is a well-instrumented **research/demo stack** with a coherent design, not the closed-loop self-improving trainer the framing implies. The genuine, working intelligence is the persisted `CheapPrior` search; the rest is either observational, confounded, or disconnected.

---

## 2. How it actually works — the real process points

There are **two loops that never connect**, plus a training substrate and a dormant router.

### 2A. The training substrate (solid)
```
data/tinyshakespeare.txt
  → prepare()              data.py:21   tokenize IF bins absent
    → tokenizer()          data.py:81   GPT-2 BPE *or byte fallback* (byte, in practice)
    → uint16 train/val.bin data.py:43
  → Loader.batch()         data.py:73   random windows, unseeded
  → LLM.forward(x,y)       model.py:249 tok_emb → blocks(GQA+RoPE+SwiGLU, pre-norm) → lm_head → CE
  → (loss/accum).backward  train.py:149 AdamW(0.9,0.95), clip 1.0, cosine LR
  → evaluate() @100 steps  train.py:164 mean over 20 *random* val batches
  → torch.save{model,cfg}  train.py:174 ckpt.pt
```
The transformer core (GQA with `n_kv_heads`, interleaved RoPE, SwiGLU, depth-scaled init) is **correctly implemented** (`model.py:99-247`, RoPE/GQA verified correct). The SFT loop is sound.

### 2B. The search / proof loop (the real "self-learning")
```
propose candidates   arms.py:123    agent(1) + prior-ranked random fill, OR evolve's crossover/mutation
  → parallel_halving  harness.py:442 concurrent rungs, checkpoint promotion (delta steps),
                                     kill switch, dedupe, eval-cache, VOCAB CLAMP→128
  → _fixed_val_loss   harness.py:378 deterministic fixed-window eval  → val_ppl
  → promote/kill      harness.py:557 keep len//eta; killed→inf→dropped
  → prior.add()       arms.py:119    refit CheapPrior from every rung eval (depth-aware)
  → prior.save()      hyperband:107  → prior_store.json  ← the ONLY compounding memory
  → make_report       arms.py:211    paired-per-seed "steps to reach random's final quality"
```
`CheapPrior` (`hyperband.py:44-129`) is an RBF-kernel weighted average over 7 featurized config dims → LCB acquisition `mean − 1.5·unc`, refit online, persisted as raw `(cfg,ppl,steps)` triples (currently 63 at version 2). This is the component that genuinely learns across runs.

### 2C. The agent audit loop (does NOT close)
```
OrchestratorAgent.step()  agents.py:356
  Trainer.propose(history) → a NEW config          agents.py:360
  train_fn(cfg,lr)         → IGNORES cfg/lr, returns hyperband winner's eval   runner:178-181
  Evaluator → sample_quality 1-10                  agents.py:366
  Judge → accept/confidence/flagged                agents.py:367
  MetaJudge → bias audit over judge_log            agents.py:382
  Human → escalate if bias OR confidence<0.4       agents.py:390  (write-only /tmp queue, never consumed)
```
Every verdict is written to `history` and a `/tmp` snapshot, but the **next round reads back only `name/config/val_ppl/params_m`** (`runner:145-147,175-177`). No verdict, bias flag, or escalation influences what is proposed or kept. The loop is observational.

### 2D. The flywheel (separate, single-shot A/B is sound)
```
train generator → generate from real prefixes → filter (heuristics OR Claude 3-judge)
  → mix into real subset at ratio → paired-train real-only vs mixed (IDENTICAL init seed)
  → deterministic real-val eval → verdict gain/neutral/collapse
```
`flywheel.py:211-300`. The single-shot A/B is genuinely controlled (paired `torch_seed`, fixed-window eval, kill switch off). The **iterated** accumulate-vs-replace comparison is not (see §3).

### 2E. The serving / onramp surface (dormant + insecure)
- **`server.py` (:8000)** — training dashboard, in-process PyTorch model singleton, can spawn training subprocesses and run arbitrary experiment configs.
- **`server_v3.py` (:8001)** — agent dashboard, reads `/tmp` snapshots + Braintrust.
- **model-onramp** — a library+CLI router (onboard→probe→route→promote) that is **inert** (`.onramp/` absent) and, when active, **bypassed** by the direct-SDK agent calls.

**Persistence map:** durable learning = `prior_store.json` + `runs/cache/` + `harness_profile.json`. Everything agent-side lives in **`/tmp`** (ephemeral). Onramp state = `.onramp/*.json` (absent). `experiments.json` = architecture sweep results.

---

## 3. Key issues, ranked by severity

### TIER 1 — invalidate a headline claim or are security-critical

**1.1 — [CONFIRMED] The arms "speedup vs random" is invalidated by a cross-arm vocab mismatch.**
`parallel_halving` clamps `vocab_size` to `effective_vocab=128` by default (`harness.py:456,480`), so the **hyperband** and **prior** arms eval at vocab 128. But **random** (`arms.py` `run_random`, no clamp) and **evolve** (`evolve.py:94` hardcodes `vocab_size:50304`) eval at 50304. `val_ppl = exp(cross_entropy)` scales with the softmax denominator, so identical weights score **~55× different** (measured 1185.95 vs 21.37). The regret target is random's ~300-scale final; hyperband hits ~17 on its *first* eval — so its "steps to random's quality" and speedup are a scale artifact, not a search win. **Fix: clamp all arms identically** (call `clamp_vocab` in `run_random` and `evolve.evaluate`, or set `auto_vocab=False` in the arms path). *Note: the `evolve` arm shares random's 50304 scale, so it is ironically the only arm with an honest regret comparison today.*

**1.2 — [CONFIRMED] The pipeline runs byte-level tokenization while claiming GPT-2 BPE.**
`data.py:81-86` silently falls back to `ByteTokenizer` (256 vocab) on any tiktoken failure. The on-disk bins prove it (ids 10–122, ~65 unique; winner `val_loss 5.06 < ln(256)`). Consequences: every reported perplexity is **per-byte, not per-token** and not comparable to any GPT-2 ppl; ~25.8M of the 50304-row embedding is dead weight; and the GRPO cloze reward degenerates to "did it predict a space?" (`grpo.py:66-67`). Worse, the tokenizer is re-decided at eval/inference time from live network state and is **not recorded in the checkpoint** — a host with different connectivity will BPE-encode prompts for a byte-trained model and silently produce garbage (`data.py:37,81`; `inference.py:79`).

**1.3 — [CONFIRMED] The agent hierarchy trains nothing it proposes and closes no loop.**
`Trainer.propose` returns a config (`agents.py:360`) that the runner's `train_fn` lambda discards, returning Hyperband's precomputed winner instead (`runner:178-181`); the adapter that *would* train the proposal is dead code (`runner:91-97`). Verdicts/bias/escalations are persisted but never read back (§2C). The "LLM-judged, human-in-the-loop gating" is unmet.

**1.4 — [CONFIRMED] The onramp self-learning routing loop is structurally open.**
Agents call the Anthropic SDK directly (`agents.py:137`), bypassing `OnrampClient`'s breaker/retry/failover/cost-cap/**exploration** (`client.py:64-127`, imported by nothing outside tests). `resolve_model` returns `candidates(role)[0]` and ranking is stable-first (`onramp_bridge.py:60`, `routing.py:76-77`), so a fresh candidate gets **zero** live traffic → `calls < min_calls(25)` → `autopilot` never promotes it (`autopilot.py:57-59`). Autopilot also has **no scheduler** (only a manual CLI), and `.onramp/` doesn't exist, so the router is inert in the checked-out tree.

**1.5 — [CONFIRMED] The deployment surface has no security boundary.**
All three servers bind `0.0.0.0` with **no auth** (`server.py:418`, `server_v3.py:73`, `dashboard.py:126`); the README even suggests public tunneling. `POST /api/experiment/run` feeds attacker-controlled kwargs into `ModelConfig(**cfg)` → `LLM(cfg)` with no bounds → **trivial memory-exhaustion DoS** (`server.py:229,273`); `/api/train/start` spawns subprocesses; `/api/agent` spends real API money on any input; and `torch.load(..., weights_only=False)` (`server.py:45`, `inference.py:60`) is **pickle-RCE** on any untrusted checkpoint.

### TIER 2 — misleading or confounded measurements

**2.1 — [CONFIRMED] The flywheel "collapse fix" confounds two variables and shows no collapse.** The accumulate report has no `ppl_gate`; the replace report has `ppl_gate:true` — so "+1.15%→+0.49%" changes both the mixing mode **and** the gate (`flywheel.py:315-317`). Both runs are `verdict:"stable"` (drift < 2%): there is small positive loss creep, **not** collapse. To isolate the effect you need the two missing cells (replace-without-gate, accumulate-with-gate).

**2.2 — [CONFIRMED] Statistical power is too low for the verdict granularity.** Single-shot uses **n=2 seeds**; in `flywheel_report.json`, `mix@0.2` flips sign across seeds (better on seed0, worse on seed1). A `gain`/`collapse` verdict at n=2 with a ±2% threshold and no variance test is unreliable (`flywheel.py:279-284`).

**2.3 — [CONFIRMED] `num_params()` under-reports by the full embedding table.** For tied embeddings it double-subtracts the shared matrix (`model.py:243-247`): the winner reports **11.3M** but actually has **37.0M** (25.8M embedding params hidden). Every "Params: X M" line is non-embedding params mislabeled as total.

**2.4 — [CONFIRMED] `experiments.py` violates its own "fair comparison" claim.** It evals on **random** val batches (`:87-88`), not the deterministic window, and `manual_seed` is set once before init, so different-sized configs consume different RNG and see **different data streams** (`:99`). Winner selection can flip on noise.

**2.5 — [CONFIRMED] The harness's headline cache benefit describes a workload the arms never run.** The 99.8×/warm-sweep numbers require identical re-sweeps, but the arms deliberately turn the cache **off** for honesty (`arms.py:140-143` vs `harness.py:8-10`). Real wall-clock savings, but not on the shipped comparison.

### TIER 3 — correctness / quality, non-blocking

- **[CONFIRMED] GRPO is un-clipped REINFORCE.** `clip_eps=0.2` is never referenced; sampled log-probs are recomputed with no importance ratio; a global `cloze_acc` term cancels under group-relative advantage (`grpo.py:42,91-93,197-198`).
- **[CONFIRMED] Speculative decoding is not distribution-exact** — resamples from `p_t` instead of the `(p_t−p_d)+` residual, by its own comment (`speculative.py:120-124`).
- **[CONFIRMED] CheapPrior features are unnormalized** — raw `d_ff_mult`/`tie` dominate the RBF kernel while `d_model`/`n_layers` are comparatively invisible (`hyperband.py:62-72`), weakening acquisition rankings.
- **[CONFIRMED] Training init is unseeded** in `parallel_halving`/`run_random`/`evolve.evaluate` — cold runs non-reproducible; the eval cache silently replays one init's number as if deterministic.
- **[CONFIRMED] Offline Judge confidence is pinned at 0.5** (a hardcoded `cloze_accuracy:0.0` makes a flag always fire, killing the high-confidence branch), so offline confidence-based human escalation never triggers (`runner:82`, `offline_agents.py:183-200`).
- **[CONFIRMED] `d_model % n_heads` and `head_dim` parity are not asserted** — `ModelConfig(d_model=384,n_heads=5)` silently truncates 4 dims and would crash RoPE if odd (`model.py:113,92`).
- **[SUSPECTED/edge] evolve gen-0 is evaluated with no budget check** (`evolve.py:226-234`) — overspends the equal-budget contract if `budget < pop*full_steps` (not triggered by shipped configs).

### Dead / half-wired code inventory
`grpo.py`, `reasoning_pipeline.py`, `speculative.py` (standalone, referenced only in `presentation.html`); `inference.py` (imported nowhere; `TensorRTLLMEngine.generate` raises `NotImplementedError`, INT4 is a comment); `synthetic_flywheel.py`'s rewrite path (`run_flywheel`/`CritiqueRewriter` unused by the A/B); `trainer_callable_for_agents` + `standard_brackets` import (`runner:91,25`); `OnrampClient` reliability stack (unused outside tests).

---

## 4. What is genuinely solid

- **The transformer** — GQA + interleaved RoPE + SwiGLU + pre-norm + depth-scaled init, correctly implemented (`model.py`).
- **The SFT training step** — grad-accum, cosine LR, clip, device/precision selection (`train.py`).
- **The throughput harness mechanics** — parallel rungs, checkpoint promotion (real delta-step savings), divergence kill, dedupe, content-addressed eval cache, vocab clamp, and CPU autotune all work as described (`harness.py`).
- **The two historically-cited measurement bugs ARE fixed** — deterministic fixed-window eval, and per-seed paired regret targets (verified).
- **The single-shot flywheel A/B** — methodologically controlled (paired init, deterministic real-val eval).
- **The model-onramp design** — coherent, and unit-tested in isolation (42 tests: probes, routing, breaker, autopilot, drift).
- **The new `evolve` arm** — genome/fitness/selection/crossover/mutation/elitism are sound; elitism is valid *because* eval is deterministic; and it's the one arm currently on an honest ppl scale.

---

## 5. Deployment ("employment") — honest assessment + path

**As-is this is a single-node research/demo stack, not deployable.** The gaps, concretely:

| Dimension | State | What's needed |
|---|---|---|
| **Security** | No auth, `0.0.0.0`, RCE via pickle load, DoS via unbounded `ModelConfig`, paid-API endpoints open | Authenticating reverse proxy + TLS; remove/admin-gate write endpoints; bound all `ModelConfig` fields; `weights_only=True` + trusted-checkpoint provenance |
| **Persistence** | Agent state in `/tmp` (ephemeral); onramp/prior are flat JSON with last-writer-wins, no cross-process lock | SQLite-with-locking (min) or Postgres for prior/stats/events/human-queue; atomic writes |
| **The self-learning loop** | Open (agents bypass `OnrampClient`; audit tier feeds nothing back; no scheduler) | Route agent calls through `OnrampClient.chat`; wire verdicts→proposal inputs; add a cron/CI for autopilot+drift+re-probe |
| **Inference/scale** | One in-process model under a global lock; fast backends unused; TRT stub raises | Dedicated inference service (finish the vLLM path); job queue for training runs; stateless API tier |
| **Interpretability** | Byte-level ppl labeled as BPE; tokenizer unrecorded | Persist tokenizer identity + effective vocab with `cfg`; assert match at load |
| **Ops** | No Dockerfile, no health checks, no metrics, config scattered across env vars | Container image, `/health`, Prometheus metrics, validated config schema |

**A staged path that respects what already works:**
1. **Make the measurements honest first** (cheap, high-value): fix the arms vocab clamp (1.1), record + assert the tokenizer (1.2), fix `num_params` (2.3), add the missing flywheel cells + more seeds (2.1/2.2). Until these land, no headline number should be quoted.
2. **Close one loop for real, end to end**: either wire verdicts back into proposal/keep decisions (1.3), or route agents through `OnrampClient` and add the scheduler (1.4). Pick one; prove it with a measured before/after.
3. **Draw a security boundary** before anything is network-exposed (1.5): auth, bound configs, safe checkpoint loading.
4. **Then** productionize persistence + inference + ops.

---

## 6. Recommended next actions, ranked

1. **Fix the arms vocab mismatch (1.1).** One-line clamp in `run_random` + `evolve.evaluate` (or `auto_vocab=False`). This is the single change that makes any current regret number trustworthy — and it directly de-risks the `evolve` arm's own comparison. *Highest value / lowest cost.*
2. **Record + assert the tokenizer (1.2).** Persist tokenizer id + effective vocab in the checkpoint `cfg`; assert on load. Then re-label ppl as per-byte or switch to real BPE. Everything downstream (ppl comparability, GRPO reward) depends on this.
3. **Decide the agent loop's fate (1.3).** Either wire the audit tier into control flow (train the proposal, feed verdicts back) or relabel it honestly as observability. As-is the framing overstates the system.
4. **Isolate the flywheel claim (2.1/2.2).** Run the 2 missing cells at ≥5 seeds before repeating "+0.49%."
5. **Security pass before exposure (1.5).** Non-negotiable if any server is reachable.
6. **Fix `num_params` (2.3)** and add the `d_model % n_heads` assertion (Tier 3) — trivial, prevent future confusion/crashes.

*Items 1, 3, and 6 are directly in scope of the search/arms work already in flight and could ship as focused follow-ups to the evolve PR.*
