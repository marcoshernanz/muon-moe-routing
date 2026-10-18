# The balance-loss block (2026-09-28)

Scripts the assistant wrote to check, run and watch the block of the plan's Amendment 5: (a), (b) and (c) at the mechanism tier with DeepSeek-V3's sequence-wise balance loss at 1e-4 beside the bias, on RQ3's seeds 3, 4 and 5, each run read against RQ3's run of its arm and seed, which differs from it by the loss alone. The decision is `context/decision/05-balance-loss-block.md`, the checks are in `context/decision/EVIDENCE-LOG.md` section 22, and the block itself is `tfg.program balance_loss` in the thesis code, on the tag `frozen-2026-09-28`.

Run the laptop's scripts from `code/` (`init_ratio.py` with `PYTHONPATH=.`), the host's from `~/code` on the host.

## Scripts

| Script | Where | What it does | Output |
|---|---|---|---|
| `init_ratio.py` | laptop | How hard the loss pulls at initialization: the thesis model at the mechanism tier, one batch of 8 real sequences, CPU float32, seeds 3 to 5; per mixture the loss and its router gradient at 1e-4 over the cross-entropy's, and the same over every parameter. | `outputs/init_ratio.txt` |
| `baseline.sh`, `baseline.py` | laptop | Whether the change leaves a run without the loss as it was: the last commit's code and the working tree train the same laptop-tier all-Muon run, 40 steps, with the first mixture recomputed, eager and compiled; every log record (the time per step aside), every saved tensor and the settings compared. | `outputs/baseline.txt` |
| `e2e.sh` | laptop | The block end to end at the laptop tier, 200 steps a run: the references trained by hand, `tfg.program balance_loss`, the references linked in, the analysis; the five claims of Amendment 5 must be decided. | `outputs/e2e.txt` |
| `upload.sh` | laptop | The code of the tag (git archive), `host.sh` and `check.py`, and the check's reference (RQ3's all-Muon run at seed 3: its settings and the first 25 lines of each log) into `~/code` on the host. | printed |
| `host.sh` | host | A guard that stops the instance at the cap, a second when no run writes its log for an hour, the cards every minute, the environment, every shard of the data, the panel; the check; one worker per card on the block, up to three rounds; the panel and the annex over the block's runs. | `logs/` |
| `check.py` | host | Whether this code without the loss trains RQ3's all-Muon run at seed 3 as RQ3 did, bit for bit, to step 250. Does not stop the block. | `logs/check.txt` |
| `watch.sh` | laptop | The host's phases, each run's step and the cards every five minutes; exits on the end of setup, of the check, of a run, of everything, on an error, an unreachable host or a heartbeat. | `outputs/watch.log` |
| `fetch.sh` | laptop | The block's run directories without their saves into `code/runs/balance_loss`, the host's logs into `outputs/host-logs/`, the check's run into `outputs/check/`. | printed |
| `link.sh` | laptop | RQ3's nine runs of (a), (b) and (c) at seeds 3 to 5 linked into `code/runs/balance_loss`, so that `tfg.analysis --runs-dir runs/balance_loss --reference adamw` pairs each run with its reference. | printed |

## Where it stands

Prepared on 2026-09-28. The laptop's checks all passed: `init_ratio.py` puts the loss's router gradient at 0.09 to 0.6 percent of the cross-entropy's at initialization, `baseline.sh` reads SAME, and `e2e.sh` decides all five claims (`EVIDENCE-LOG.md` section 22). Marcos gave the go-ahead the same day, with a 25-dollar cap on Vast.ai; the code the host runs is the tag `frozen-2026-09-28`.

Ran on 2026-09-28 and 29 on a Vast.ai host of four RTX 5090s (instance 53255222, host 120872 in Taiwan): setup 7 minutes, the check IDENTICAL, the 9 runs in one round of four workers in 5 h 01 min, the evaluation in 3 minutes, 11.84 dollars, the instance destroyed at 02:30 UTC with none left (`outputs/instance.log`, `outputs/watch.log`, `outputs/host-logs/`, the check's run in `outputs/check/`). The runs are in `code/runs/balance_loss/` and on the Hub under `balance_loss/`; run `link.sh` before the analysis. `S-loss`, `G-loss`, `O-loss` and `L-loss-router` pass, `L-loss` reads `direction` (`EVIDENCE-LOG.md` section 23; the plan's Amendment 6).
