# The geometry pilot (2026-09-24)

A one-seed pilot the assistant ran after Gate 1 failed, to decide the second registration of `thesis/PLAN.md` before the second chain. It asks two things. Does Muon's higher bias span in the first mixture come from the router's own optimizer or from the layers that feed the router? And do the new readings of the router's input geometry (`tfg/telemetry.py`: `router_input_similarity`, `mean_score_span`, `raw_usage_perplexity`, `raw_max_load_ratio`, `router_update_common_share`) leave training bit for bit as it was? The results are in `context/decision/EVIDENCE-LOG.md` section 17.

Every run uses the tuning block's settings: the mechanism tier, 0.5B tokens in 7,629 steps, seed 0, a cosine schedule, compiled, one recomputed block, at the tuned rates (AdamW 2e-3, Muon 4e-3). The working tree's code of the day was copied to each host, not a tag. Two of the runs repeat a tuning run of the first chain with the new readings on: (a) at 2e-3 and (b) at 4e-3.

## Hosts

Three one-card RTX 5090 hosts on Vast.ai, each at 575 W and 120 GB of disk:
- **Host A** (Ukraine, 0.567 dollars an hour) downloaded the locked environment at under 1 MB/s and was destroyed after 33 minutes, before it ran anything.
- **Host B** (South Korea, 0.633; driver 580.159.03) ran the Muon arms.
- **Host C** (California, 0.573; driver 580.126.09) replaced A and ran the suite on its card, then the AdamW reference and the arms with the backbone's groups on Muon.

Runs were cut and one added to keep the pilot inside its 5 dollars:
- The two no-balancer arms were skipped: `pilot.sh`'s lists still name them, and a guard on each host ended them the moment they started (`skipped.txt`).
- (d) moved from host C's list to host B's and was stopped at step 3,010 (`experts.sh`). The script then hung: its `wait` also waited for the card logger, which never ends, so it stamped no end and touched no `logs/done`, and host B sat idle from about 16:00 to 16:44 UTC, when it was destroyed (about 0.45 dollars).
- (h)'s first 860 steps left the router's inputs as dissimilar as all-AdamW's, while all-Muon's and (c)'s were four times higher by then. So (h) was stopped at step 3,000 like (d), and its place on host C went to (i), Muon on attention and the dense block together, every matrix that feeds the first mixture (the `muon_attention_dense` partition, added to `tfg/optim.py` for it), also stopped at step 3,000 (`feeding.sh`, which waits for the training process alone).

The first mixture's readings separate the arms by step 3,000.

## Scripts

| Script | Where | What it does | Output |
|---|---|---|---|
| `setup.sh` | host | The RUNBOOK's sections 1 and 2 on a one-card host, without the panel: the card, CPU, memory and disk, `uv sync --frozen`, every shard of the data, each timed. | `logs/setup.txt` |
| `pilot.sh` | host | The host's list of runs, one after another on its card (`pilot.sh A` on host C, `pilot.sh B` on host B); host C first runs the suite. Logs the card every 30 s. | `logs/pilot.log`, `logs/phases.txt`, `logs/gpu.csv`, `logs/pytest.txt` |
| `experts.sh` | host B | (d) after host B's list, stopped once its routing log passes step 3,000. | appends to the same logs |
| `probe.sh` | host | One look for `watch.sh`: the phases, the runs with their last step and loss, new log lines naming a problem, the card, the free disk. | printed |
| `feeding.sh` | host C | Stops (h) once its routing log passes step 3,000, waits for `pilot.sh`'s list to end, then trains (i) and stops it at step 3,000. | appends to the same logs |
| `watch.sh` | laptop | `probe.sh` on both hosts every five minutes; exits on a problem, a new phase, both hosts done, a host unreachable three times, a first run passing step 1,000, or a two-hour heartbeat. | `outputs/watch.log` |
| `compare.py` | laptop, from `code/` with `PYTHONPATH=.` | The two reruns against the first chain's runs, every logged number step by step; then every run's endpoints (all the mixtures and the first alone, integrated, over the warmup and early), with the first chain's Muon run at 2e-3 beside them, over each whole run and over steps up to 3,000. | `outputs/bit-for-bit.txt`, `outputs/pilot.csv`, `outputs/pilot-3000.csv` |
| `windows.py` | laptop, from `code/` with `PYTHONPATH=.` | The first mixture's span, router-input cosine, mean score span and raw max-load ratio per arm by windows of 500 steps; the cosine per mixture late in the window every arm has; the lowest bias in (d), (b) and (a) against the rest; the stress grid's (b) and (c) at 1x by window, three seeds. | `outputs/windows.txt` |
| `chain2.py` | laptop | The second chain's wall time and cost on a host of four 5090s, with the model of `host_checks_2026_09/chains.py`, with and without (i), and RQ3 and the confirmation tier each alone. | `outputs/chain2-cost.txt` |

## Outputs

`outputs/host-B/` and `outputs/host-C/` are each host's `~/code/logs/` when its list was over: `setup.txt`, `pytest.txt` (host C), `pilot.log`, `phases.txt`, `gpu.csv` and `skipped.txt`, with `logs-setup.out`. Host B's `phases.txt` ends at (d)'s start, for the reason above. The runs themselves, without their saves, are in `code/runs/pilot/` (not in git) and in the private Hub dataset `marcoshernanz/tfg-runs` (24 files, 150 MB, uploaded 2026-09-24 and checked byte for byte by downloading it back).

## Results

**Bit for bit.** (a) and (b) match the first chain's tuning runs at the same rates on every number both logged, all 763 lines of both logs (`outputs/bit-for-bit.txt`). The new readings leave training as it was on the card.

**The first mixture, steps 501 to 3,000**, the window every arm has (`outputs/windows.txt`, `outputs/pilot-3000.csv`):

| Arm | Muon on | Bias span | Router-input cosine | Mean score span | Raw max-load ratio | Span, steps 2,501 to 3,000 |
|---|---|---:|---:|---:|---:|---:|
| (a) | nothing | 0.123 | 0.073 | 0.098 | 1.87 | 0.086 |
| (g) | the router | 0.137 | 0.068 | 0.138 | 1.74 | 0.128 |
| (h) | the dense block 0 | 0.176 | 0.040 | 0.139 | 2.36 | 0.123 |
| (e) | attention | 0.250 | 0.062 | 0.189 | 2.57 | 0.143 |
| (i) | attention and the dense block 0 | 0.583 | 0.063 | 0.481 | 6.65 | 0.289 |
| (d) | the experts | 0.776 | 0.143 | 0.618 | 7.49 | 0.727 |
| (c) | everything but the router | 0.372 | 0.430 | 0.271 | 5.73 | 0.131 |
| (b) | everything | 0.423 | 0.434 | 0.273 | 4.17 | 0.255 |

What it says:
- **The shared direction needs the whole backbone on Muon.** The router's inputs share a direction under (b) and (c), 0.43, and under no single part: (d) reaches 0.14 to 0.16, and (e), (h) and (i) stay at or below all-AdamW's 0.07. Under (b) and (c) the cosine falls with depth, from 0.41 in the first mixture to 0.09 in the seventh; no part on Muon moves it to another mixture.
- **The router's own optimizer does not raise the span.** (g) reads like (a) on every reading, and (c), with the router on AdamW, reads like (b) on the cosine.
- **Parts of the backbone on Muon raise the span by another route.** Without the shared direction, a few experts get a large offset in the router's scores (the mean score span and the raw max-load ratio), which the bias cancels. In (d) it lasts and is one expert: over steps 2,001 to 3,000 one bias sits at -0.43 against the others' +0.14, and the span without that expert is 0.29 of 0.74. In (i) it is large and falling: 0.79 over steps 501 to 1,000, 0.29 over 2,501 to 3,000, still above (b)'s 0.26 and (a)'s 0.09. In (e) it passes: 0.51 after the warmup, 0.14 by step 3,000. (h) stays near (a).
- **(b) against (c) late in a run depends on the seed.** In the pilot (c) falls below (b) after step 1,500 (0.13 against 0.26 over steps 2,501 to 3,000). In the stress grid at 1x, over steps 4,001 to 7,629, (c) is below (b) at seed 0 (0.147 against 0.247) and above it at seeds 1 and 2 (0.290 against 0.192, 0.182 against 0.163).
- **The warmup's churn** over all the mixtures is highest with the router on AdamW behind a Muon backbone, as in the stress grid: (c) 0.118 against (a) 0.102 and (b) 0.082.
- **C6 follows the experts' optimizer.** The correlation of an expert's update size with its load over steps up to 3,000 is 0.50 to 0.57 in every arm with the experts on AdamW ((a), (e), (g), (h), (i)) and 0.09 to 0.10 in every arm with them on Muon ((b), (c), (d)).
- **Loss.** Over whole runs, (g) ends where (a) does (3.387 against 3.386) and (c) where (b) does (3.305 against 3.300). The runs stopped at step 3,000 are at 3.72 to 3.76 there, on a schedule set for 7,629 steps, and are not compared on loss.

**What the second registration takes from it.** The source claims it had reserved (`S-source` and `G-source` on one feeding group) are replaced by two parts claims: `S-parts`, (a) below (d) and (i) on the first mixture's integrated span, the two parts whose rise lasted or stayed above all-Muon's to step 3,000; and `G-parts`, every single part ((d), (e), (h), (i)) below (b) and (c) on the router-input cosine. (i) joins RQ3 (33 runs). No direction is registered for (b) against (c). If (i)'s excess over (a) vanished at step 3,000, it would still add about +0.08 to the integrated span of a 1B-token run: (0.583 - 0.123) × 2,500 / 14,759.

**Spend.** 4.71 dollars of the 5. Host A ran from 13:00 to 13:33 UTC, host B from 13:00 to 16:44 and host C from 13:34 to 17:29, billed from when each was running. No instance is left.
