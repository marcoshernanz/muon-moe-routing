# The first chain of the program (2026-09-23)

Scripts the assistant wrote to run the RUNBOOK's first chain (section 5: the tuning, Gate 1 and the RQ2 stress grid) on a rented Vast.ai host of four RTX 5090s, one worker per card, on the code of the tag `frozen-2026-09-23`, and to watch it from the laptop. The runs themselves come home to `code/runs/` (the RUNBOOK's last section); this folder keeps the host's records and the laptop's watch log. The results are in `context/decision/EVIDENCE-LOG.md`.

Run the host's scripts from `~/code` on the host, the laptop's from anywhere.

## Scripts

| Script | Where | What it does | Output |
|---|---|---|---|
| `setup.sh` | host | The RUNBOOK's sections 1 and 2: the host's cards, power limits, CPU, memory and disk; `uv sync`; every shard of the data with the Hub's client at sixteen workers, then `download()`'s check; the panel. | `logs/setup.txt` |
| `checks.sh` | host | Muon's step at the mechanism tier on all four cards at once (`host_checks_2026_09/step.py`, 300 steps), against the 365.6 ms of EVIDENCE-LOG section 15; then the suite on card 0 (RUNBOOK section 3). | `logs/step-gpu<n>.txt`, `logs/pytest.txt` |
| `chain.sh` | host | The first chain with four workers, and `edge.py` between the tuning and Gate 1; the cards' power, use, memory, temperature and clock every 30 s, the host's memory and disk every minute. | `logs/first-gpu<n>.log`, `logs/gpus.csv`, `logs/host.txt`, `logs/phases.txt` |
| `edge.py` | host | Stops the chain before Gate 1 when AdamW's or Muon's best rate is still at an edge of its grid after the sweep's extra rate, so that Gate 1 and the stress grid do not train at a rate the grid does not bracket. | printed to the workers' logs |
| `status.py` | host | Each block's runs by status, and every run under way with its step, loss, milliseconds per step and last validation loss. | printed |
| `probe.sh` | host | One look for `watch.sh`: new log lines naming a problem, the live workers, the free disk, each card's power limit and temperature, and whether the chain is over. | printed |
| `watch.sh` | laptop | `probe.sh` over ssh every five minutes; exits on a problem, a change in the live workers, the end of the chain, a host unreachable three times, or a two-hour heartbeat. | `outputs/watch.log` |
| `per_mixture.py` | laptop | Gate 1's integrated bias span and load spread per mixture, from the runs' routing records, to see whether the reversal is one layer's. | `outputs/per-mixture.txt` |
| `rethink.py` | laptop, from `code/` with `PYTHONPATH=.` | The chain read again for the rethink of 2026-09-24: the tuning sweeps' span in the first mixture and the rest, Gate 1's span per mixture by window of training and per seed, the annex's hidden-state cosine per mixture, the stress grid's span, churn and router steps per partition and multiplier, and the T4 tier's span per mixture for comparison. | `outputs/rethink.txt` |
| `per_seed.py` | laptop, from `code/` with `PYTHONPATH=.` | The readings of the second registration seed by seed on Gate 1's two arms and the stress grid's three partitions at 1x, integrated after the warmup, over the warmup and over the early window, in all the mixtures and in the first alone, to see which of its predictions three fresh seeds could decide. | `outputs/per-seed.txt` |
| `windows.py` | laptop, from `code/` with `PYTHONPATH=.` | The first mixture's bias span by window of training in the tuning's AdamW run at 2e-3 and Muon's at 2e-3 and 4e-3 and in Gate 1's six runs, to see how much of Muon's span its rate carries and how the span changes from 0.5B tokens to 1B. | `outputs/windows.txt` |

## Outputs

`outputs/host-logs/` is the host's `~/code/logs/` as it stood when the chain was over: `setup.txt`, `step-gpu<n>.txt`, `pytest.txt`, the workers' logs `first-gpu<n>.log.gz` (gzipped, 42 MB unpacked; `zgrep` reads them), the evaluation's `eval-gpu<n>.log` (Gate 1's six runs, on the two cards the 16× level had left free), `gpus.csv` (time, card, power, limit, use, memory, temperature, clock), `host.txt`, `phases.txt`, and `deleted-tuning-saves.txt`. `outputs/stress.txt` is `tfg.stress --runs-dir runs/rq2` on the whole grid, `outputs/rethink.txt`, `outputs/per-seed.txt` and `outputs/windows.txt` the tables of `rethink.py`, `per_seed.py` and `windows.py`, `outputs/watch.log` the laptop's watch.
