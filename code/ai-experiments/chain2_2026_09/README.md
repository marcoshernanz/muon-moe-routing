# The second chain of the program (2026-09-25)

Scripts the assistant wrote to run the RUNBOOK's second chain (section 5: the RQ3 block, 33 runs on seeds 3 to 5, and its evaluation) on a rented Vast.ai host of four RTX 5090s, one worker per card, on the code of the tag `frozen-2026-09-25`, and to watch it from the laptop. The confirmation tier is not part of it. The runs themselves come home to `code/runs/rq3/` (the RUNBOOK's last section); this folder keeps the host's records, the laptop's watch log and the search for the host. The results are in `context/decision/EVIDENCE-LOG.md`.

Run the host's scripts from `~/code` on the host, `crosscheck.py` and `summary.py` from `code/` on the laptop, and the laptop's other scripts from anywhere.

## Scripts

| Script | Where | What it does | Output |
|---|---|---|---|
| `offers.py` | laptop | Polls Vast.ai for a fit host of four 5090s (driver 580 or newer, cards that reach the given watts, reliability 0.95 or more, 200 GB of disk, not in China, not the two hosts that failed) and exits when the cheapest is at or under a price. | `outputs/offers.log` |
| `upload.sh` | laptop | The code of the tag (git archive), the host scripts, the host checks' `step.py` and the tuning block's `rates.json` into `~/code` on the host. | printed |
| `setup.sh` | host | The RUNBOOK's sections 1 and 2, as for the first chain: the host's cards, power limits, CPU, memory and disk; `uv sync`; every shard of the data; the panel. | `logs/setup.txt` |
| `checks.sh` | host | Muon's step at the mechanism tier on all four cards at once (300 steps, compiled), then the suite on card 0 beside the RQ3 block end to end at 20 steps on cards 1 to 3 (trained, evaluated, analysed) into a scratch directory it removes. | `logs/step-gpu<n>.txt`, `logs/pytest.txt`, `logs/smoke-*.log`, `logs/checks.txt` |
| `chain.sh` | host | The RQ3 block with four workers, relaunched up to three rounds while a run is neither finished nor diverged, then its evaluation with four workers; the cards every 30 s and the host every minute beside it; stops the instance itself four hours after the end if nobody has fetched the results. | `logs/second-gpu<n>.log`, `logs/eval-gpu<n>.log`, `logs/gpus.csv`, `logs/host.txt`, `logs/phases.txt` |
| `status.py` | host | The block's runs by status, and every run under way with its step, loss, milliseconds per step and last validation loss. | printed |
| `probe.sh` | host | One look for `watch.sh`: new log lines naming a problem, a run newly diverged, a stall, the stage, the live workers, `status.py`, the free disk, each card's power limit and temperature, and whether the chain is over. | printed |
| `watch.sh` | laptop | `probe.sh` over ssh every five minutes; exits on a problem, a change in the live workers or the stage, the end, a host unreachable three times, or a two-hour heartbeat. | `outputs/watch.log` |
| `fetch.sh` | laptop | The block's run directories without their saves into `code/runs/rq3`, and the host's logs into `outputs/host-logs/`; at the end, gzips the workers' logs and stands down the host's safety stop. | printed |
| `crosscheck.py` | laptop | After the analysis, without `tfg`: every run's settings against its arm, one data order, complete logs, and the first mixture's integrated span and router-input cosine recomputed from the routing logs against `endpoints.csv`. | `outputs/crosscheck.txt` |
| `summary.py` | laptop | The numbers of the evidence log's section 18: the registered endpoints per run, the step time and the early and final span per arm, the pairs no claim orders, Muon at half its rate against its tuned rate, whether a rise sits on one expert, the cards and the disk. | `outputs/summary.txt` |

## What happened on the host

- The first host rented (offer 46753296, United States, host 410852, 500 W cards, 2.00 an hour) never loaded its image: its Docker daemon went through a local proxy that refused connections. It was destroyed after 10 minutes, and `offers.py` skips that host since.
- The host of the chain: instance 52639774 from offer 28862480, Taiwan, host 1801, machine 44463, four RTX 5090s at a 575 W limit (600 W maximum), driver 580.173.02, a Xeon Platinum 8352V with 144 threads, 503 GB of memory, 200 GB of disk, 2.17 dollars an hour.
- Setup: `uv sync` 306 s, the data 483 s (1,021 files, 19 GB), the panel 30 s.
- Checks: Muon's step 401.8 to 403.9 ms on the four cards at once (26.6 GiB allocated, 28.9 reserved), against 363.7 to 374.7 on the first chain's host with the first registration's code; the geometry pilot, on this code, logged 394 ms for Muon and 387 for AdamW against Gate 1's 374 and 317, which put the cost of the router-input readings the second registration added at about 5 percent under Muon and 20 under AdamW; the chain's own steps (below) put it at about 7 percent under either, host and code together. The suite: 203 passed. The smoke pass first ran eager, which does not fit the mechanism tier in 32 GB (out of memory at the first step, as the RUNBOOK's section 4 says; `smoke-*` in the host logs). Run again by hand, compiled as the chain trains, on all four cards once the suite was done (`smoke2-*`), it trained the 11 arms, evaluated all 11 on the panel and the annex, and the analysis wrote its 134 files with every claim's contrasts (S 8, S-rate 2, S-parts 4, G 8, G-parts 16, R 2, M 4, C3 2, C4 1, C5 1, C6 30, C7 24, both views counted). `checks.sh` runs it compiled since.
- The suite left 16 GB under `/tmp/pytest-of-root`, deleted before the chain so that the saves (about 135 GB) leave about 40 GB free.
- The chain started on 2026-09-25 at 20:17:46 UTC. Training took one round: every worker exited 0 on 2026-09-26 at 10:54:22 UTC (14 h 37 min), with all 33 runs finished and none diverged. The evaluation took 9 min 30 s on the four cards and wrote 33 panels and 33 annexes at batch 32.
- The runs held 175 to 39 GB of free disk. While training, the cards drew a median of 517 W at their 575 W limit (399 W at the 5th percentile), at a median of 2,820 MHz, and reached at most 83 °C (`gpus.csv`). The workers' logs name no error.
- The step: 337 to 343 ms with the experts on AdamW, whatever else is on Muon, and 394 to 409 ms with the experts on Muon, against Gate 1's 317 and 374 on the first chain's host, so the 33 runs are 51.6 card-hours: 12.9 hours on four cards packed perfectly, against the 14.6 they took, most of the difference being the last run training alone for its final hour.
- `fetch.sh final` brought the 33 run directories home (165 files, 1.5 GB, the same checksums as on the host), and the instance was destroyed on 2026-09-26 at 11:08:59 UTC, with no instance left. The runs and both analyses are in the Hub dataset `marcoshernanz/tfg-runs` (436 files under `rq3/`, the runs checked byte for byte by downloading them back).
- Spent: 33.79 dollars on Vast.ai, the first host's 10 minutes included: about 15 h 32 min of the host, of which 40 minutes of setup and checks and 14 min of evaluation and fetching after training.

## Outputs

`outputs/instance.log` records the rentals (its times are the laptop's, CEST, unless marked UTC), `outputs/offers.log` the search, `outputs/watch.log` the laptop's watch, `outputs/host-logs/` the host's `~/code/logs/` as it stood when the chain was over, and `outputs/crosscheck.txt` and `outputs/summary.txt` the checks and numbers read at home.
