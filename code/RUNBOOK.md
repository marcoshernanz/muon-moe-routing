# First day on the RTX 5090

Everything runs from `code/` with `uv`. An interrupted command is rerun as it was (a run by hand, with `--resume`; without it a run starts over in an emptied directory: the logs, checkpoint, early save, and evaluations of the run it replaces are removed): runs resume from their last checkpoint, and a sweep skips the rates it already finished. A run whose training loss or gradient norm stops being a finite number has diverged: it ends on that step, its last line in `metrics.jsonl` carries a `val_loss` of NaN, and a relaunch leaves it as it is. So does it leave a run whose log reaches its last step, its `settings.json` included, when its `checkpoint.pt` stands at that step too or has been deleted, which is how disk is freed (section 8). A run killed while it wrote its last checkpoint has a whole log and the checkpoint before: a relaunch trains its tail again, and until then `tfg.program evaluate` skips it, so that the panel never reads a model that is not the run's last. A checkpoint ahead of either log (`metrics.jsonl` or `routing.jsonl`) means records were lost (a machine that went down before the log reached the disk), and the resume is refused: train that run again, by hand without `--resume` or by deleting its directory and relaunching its block. A relaunch with other settings than the run was made with is refused once the run has a checkpoint or is over, the settings named, the logging and evaluation intervals among them (a run cut before its first checkpoint has nothing to go on from and starts over under the settings it is given); besides the device and the checkpoint interval, which change nothing a run computes, the one thing that may change is the number of steps of a run on the constant schedule, which can be trained longer from its checkpoint, by hand and one run at a time (a stress block cannot be, since its calibration run would move and its clips with it, and a run that diverged stays as it is), never cut to the step its checkpoint stands on or to one before it, and not at all once the checkpoint is gone. A run trained longer ends on the weights of an uninterrupted run of the new length; its log keeps the record of its old last step, and its `early.pt` stays where the old length put it.

## 1. Machine

The cards are RTX 5090s, one host of four rented on Vast.ai (`context/decision/COMPUTE-BUDGET.md`; `context/research/B8-h100-rental-comparison-2026-09.md` has the provider's commands). Two card models give different losses from the first step, so a comparison that starts on 5090s ends on them. Rent a verified, on-demand host whose cards run at 575 W or more (`gpu_max_power`; a capped card is slower) under a driver of 580 or newer (the lockfile's torch is built for CUDA 13), with 250 GB of disk for the third chain (200 held the second's; section 5):

```bash
vastai search offers 'num_gpus=4 gpu_name=RTX_5090 verified=true rentable=true disk_space>=250' -o dph_total
vastai create instance <offer> --image vastai/base-image:cuda-12.9-mini-py314-2026-09-07 --disk 250 --ssh --direct
vastai attach ssh <instance> "$(cat ~/.ssh/id_ed25519.pub)"
vastai show instance <instance> --raw   # public_ipaddr, and the host port mapped to 22
```

The search cannot filter on the driver; `--raw` gives each offer's `driver_version` and `gpu_max_power`. A host that still refuses the key a few minutes after the attach is destroyed and another rented (one in Japan did, twice, on 2026-09-23).

The repository is private, so the code goes up from the laptop, as the tag the runs are made with (1 MB without `ai-experiments/`; the second chain's was `frozen-2026-09-25`, the third's is `frozen-2026-09-26`), and uv is installed there:

```bash
git archive frozen-2026-09-26:code tfg pyproject.toml uv.lock RUNBOOK.md | ssh -p <port> root@<ip> 'mkdir -p code && tar x -C code'
ssh -p <port> root@<ip> 'curl -LsSf https://astral.sh/uv/install.sh | sh && cd code && ~/.local/bin/uv sync --frozen'
```

A host after the first chain's has no tuning block, and the RQ3, confirmation and balancer blocks read the tuned rates from it (they stop at once without them), so they go up too:

```bash
ssh -p <port> root@<ip> 'mkdir -p code/runs/tuning' && scp -P <port> ~/dev/tfg/code/runs/tuning/rates.json root@<ip>:code/runs/tuning/
```

The sync took 47 s on one host and 65 minutes on another, whose download of the locked PyTorch wheels was slow (`context/decision/EVIDENCE-LOG.md` section 16); the host is paid for either way.

Everything below runs in `~/code` on the host. The disk holds the environment, the data (20 GB), and every run's `checkpoint.pt` and `early.pt` until the run is evaluated or its block no longer needs them (sections 5 and 8).

The host bills by the second while it runs. Keep it running only while runs are: when a chain ends, or when the next step waits for a decision (Gate 1's reading, say), bring the logs home (last section) and stop the instance (`vastai stop instance <instance>`, which leaves only the disk billed, 0.07 to 0.26 dollars an hour at 400 GB on the hosts of 2026-09-23); destroy it (`vastai destroy instance -y <instance>`) once nothing on its disk is needed, and check that `vastai show instances` lists nothing left running. A stopped host restarts only when its cards are free, so a restart can wait.

## 2. Data

The shards are public on the Hub, so the host needs no login. Fetch all of them before the first run: a run's data order depends on how many training windows there are, which it records, and the analysis refuses a block whose runs saw different counts. `download()` fetches one file at a time, about 10 MB/s from the Hub, so the Hub's own client fetches them first, sixteen at a time (19 GB in about five minutes), and `download()` then checks that every shard the metadata counts is there:

```bash
uv run hf download marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full --repo-type dataset --local-dir data/fineweb-edu --max-workers 16
uv run python -c "from tfg.data import download; download()"
uv run python -m tfg.panel build
```

## 3. Checks

```bash
uv run pytest -q
```

On CUDA the grouped-dispatch test compares the `grouped_mm` path against the per-expert loop on the GPU in bf16, so a green suite here is the Gate 1a "grouped dispatch matches the reference" item. Checkpoint and resume are covered by the same suite.

Runs are deterministic: `train` turns on `torch.use_deterministic_algorithms`, so the same seed gives the same numbers and a resumed run matches an uninterrupted one bit for bit, which the suite checks. That mode raises on any kernel without a deterministic path. None was hit on a T4, the H100 or the 5090; if a card hits one, the message names it, and the fallback is `warn_only=True` on that line in `tfg/train.py`, which keeps training and prints the kernel instead.

## 4. Throughput

```bash
uv run python -m tfg.train --tier mechanism --partition muon --steps 60 --warmup-steps 10 --compiled --recompute 1 \
  --log-every 10 --eval-every 60 --checkpoint-every 60 --run-dir runs/throughput
```

The short warmup is there because the cosine schedule refuses a run no longer than its warmup, 500 steps by default. `--compiled` runs the training forward through `torch.compile`; the first step takes a minute to compile, and the flag is part of a run's settings, so every real run below carries it. Read `tokens_per_s` from the log lines after the first one. One mechanism-tier step is 65,536 tokens, so a 0.5B-token sweep run is 7,600 steps and a 1B-token gate run is 15,300 steps; divide by the measured rate to budget the day. Watch `nvidia-smi` for utilization while it runs; `torch.compile` is the first lever if it is low. Run the command once with `TORCH_LOGS=graph_breaks` in front: on a card that runs `grouped_mm` the model compiles as one graph and the log stays silent (checked on the H100 and the 5090; with `TORCH_LOGS=graph_breaks,recompiles` one recompilation shows, the evaluation's forward without gradients), and a line there names the statement that split the graph and left part of the model uncompiled (a T4 always prints some, because it runs the experts slice by slice).

The 5090 holds 31.4 GiB, and both tiers need `--recompute K`, which keeps only the input of the first K mixture blocks and computes them again in the backward pass: the same run to rounding under `--compiled` and bit for bit without it (the suite checks both). On a 5090 with the speed patches (`context/decision/EVIDENCE-LOG.md` section 13) Muon's peak at the mechanism tier is 30.4 GiB without it and 27.7 at K = 1, and at the confirmation tier 30.0 at K = 8 and 28.0 at K = 9. On a logged step the telemetry also holds a copy of every parameter at the peak, about 1.2 GiB at the mechanism tier and 1.9 at the confirmation tier, hence K = 1 and K = 9. With the telemetry on and all four cards of a host training at once (2026-09-23), the peaks at the mechanism tier were 26.6 GiB allocated (28.9 reserved) under Muon and 27.7 (29.7) under AdamW, and at the confirmation tier 26.5 (30.0) and 28.3 (30.3); a worker empties the cache before each run, a sweep's runs included, so its second run peaks where its first did (before a sweep did so, on 2026-09-23, a worker's second AdamW tuning run started beside the cache the one before left, logged the allocator's `expandable_segments` warning twice and trained on). Check both on a new host: the command above, and the same with `--tier confirmation --recompute 9 --run-dir runs/memory`, reading the peak off `nvidia-smi` while it runs (`torch.cuda.max_memory_allocated` is not logged). K is part of a run's settings, so a block carries one K on every run, as the commands below do. The annex runs a backward pass without recomputation, which does not fit at the tier's batch of 64 at either tier: `evaluate` runs at 32 at the mechanism tier (25.5 GiB) and at 16 at the confirmation tier (section 5).

## 5. The program (Gate 1a onward)

Every block of the plan is one command. A block trains its runs one after another into `runs/<block>/`, resumes a run that was cut and skips one that finished, so after a preemption the same command continues where it stopped; a run whose settings changed since it was made (other rates, say, or the compiler flag: give every command of a block the same one) is refused rather than reused, with the changed settings named, so move it away first; only a run cut before its first checkpoint starts over under the new ones. Steps follow from the plan's token counts (1B tokens is 15,259 steps at the mechanism tier). Every run is compiled unless `--no-compiled`.

```bash
uv run python -m tfg.program tuning --recompute 1
```

Gate 1a: AdamW's five rates on the all-AdamW partition, then Muon's five on the Muon partition with AdamW's tables and norms at the rate that won, then the momentum control's six on its own partition the same way, 0.5B tokens each, one seed, then the AdamW check run at β2 0.98. The control's sweep tunes nothing, since no later block runs the control; it is the material of the frontier (section 7), and `--control-rates ...` widens it. Runs warm up and then decay the rate on a cosine to a tenth of the peak by the last step, so the sweep compares rates after decay, as the plan asks. The winners go to `runs/tuning/rates.json`, which the later blocks read; `--adamw-lr` and `--muon-lr` stand in for it. A rate whose run diverged cannot win. If a winner sits at the edge of its grid, the sweep trains one more rate a grid step past that edge, as the plan asks, and says so; `sweep.json` records it as `extension`, and `edge` tells whether the winner is still at an edge after it. When AdamW's or Muon's is, the block stops there, before the next sweep and before `rates.json`, so that a chain goes no further: widen that grid by hand (`--adamw-rates ...` or `--muon-rates ...`) and rerun, and the finished rates are skipped. The control's is only reported, since no later block reads it.

```bash
uv run python -m tfg.program gate1 --recompute 1
uv run python -m tfg.analysis --runs-dir runs/rq1 --reference adamw
```

Gate 1: partitions (a) all AdamW and (b) Muon everywhere except tables and norms, three seeds, 1B tokens, into `runs/rq1`. The plan's first registration decided the gate on the sign of `bias_span` across these seeds; it ran on 2026-09-23 and 24 and failed, Muon's span above AdamW's at every seed (the plan's section 13). The analysis (section 7) reads the second registration's claims that (a) and (b) allow on these runs, as a pilot of the RQ3 block.

```bash
uv run python -m tfg.program rq3 --recompute 1
uv run python -m tfg.program evaluate --block rq3 --batch-size 32
```

The RQ3 block, into `runs/rq3`, on seeds 3, 4 and 5, which no earlier run had, at 1B tokens on the cosine schedule: seed by seed, so that every arm has one seed before any has two, the partitions (a) `adamw`, (b) `muon`, (c) `muon_except_router`, (d) `muon_experts`, (e) `muon_attention`, (g) `muon_router`, (h) `muon_dense` and (i) `muon_attention_dense`, then (b) with Muon's rate brought down to AdamW's (`--muon-lr-multiplier`, named `muon-muonlr0.5x` at the tuned rates of 2e-3 and 4e-3, and left out when the two rates are equal), then (a) and (b) without the balancer: 33 runs. `evaluate` runs the panel and the annex (section 8) over every finished run of the block, on the machine that holds the checkpoints.

```bash
uv run python -m tfg.program rq2 --recompute 1
uv run python -m tfg.stress --runs-dir runs/rq2
```

The stress grid (section 9): per partition (a), (b), (c), a calibration run at 1x on the constant schedule under `runs/rq2/calibration/`, then the family at 1x, 2x, 4x, and 8x, seed by seed, with both clips at twice that run's post-warmup median norms (the block stops with an error if that run diverged or did not reach its last step, since then there is nothing to set the clips from), and on (b) the balancer-speed control at 4x, trained only when 4x is one of the multipliers, beside the arm it is read against; last, for a partition none of whose runs at the highest multiplier diverged, the family at 16x (`--extension None` leaves it out). Its optional confirmation-tier subset is `--tier confirmation --tokens 1000000000 --multipliers 2 4 --extension None` with `--runs-dir` somewhere of its own, `--adamw-lr` and `--muon-lr` at the re-checked rates of `runs/confirmation/recheck/rates.json`, and `--recompute 9`: 24 runs.

```bash
uv run python -m tfg.program confirmation --recompute 9
uv run python -m tfg.program evaluate --block confirmation --batch-size 16
uv run python -m tfg.program balancer --recompute 1
uv run python -m tfg.program evaluate --block balancer --batch-size 32
```

The confirmation tier: an eighth, a quarter, a half and all of each tuned rate re-checked at 1B tokens under `runs/confirmation/recheck/` (at and below the mechanism tier's optimum, since the best rate falls as a run gets longer), each sweep with one more rate past a winning edge as in the tuning block, then (a), (b), (c) at three seeds and 3B tokens at 0.7 times the rates that won, the fall of the best rate from 1B to 3B tokens (Amendment 3). The balancer block, into `runs/balancer`, on seeds 6, 7 and 8, which no earlier run had, at 1B tokens at the mechanism tier: seed by seed, (a) and (b) with the bias at its rate, then both with it four times faster (`--balancer-multiplier 4`, named `adamw-balancer4x` and `muon-balancer4x`): 12 runs. Its (a) and (b) read the claims on that pair once more, on seeds of their own, C4 among them, hence its evaluation.

```bash
uv run python -m tfg.program balance_loss --recompute 1
uv run python -m tfg.program evaluate --block balance_loss --batch-size 32
```

The balance-loss block of Amendment 5, into `runs/balance_loss`, on RQ3's seeds 3, 4 and 5 at 1B tokens at the mechanism tier: seed by seed, (a), (b) and (c) with DeepSeek-V3's sequence-wise balance loss at 1e-4 beside the bias (named `adamw-balanceloss0.0001` and so on): 9 runs. It does not train its references: they are RQ3's runs of the same arms and seeds, which differ from its runs by the loss alone, and at home `ai-experiments/balance_loss_2026_09/link.sh` links them into `runs/balance_loss` before the analysis (`--runs-dir runs/balance_loss --reference adamw`). It runs as a chain of its own, with that folder's scripts (its README), whose host script first checks on the host that the code without the loss trains RQ3's run of (b) at seed 3 as RQ3 did.

Every training block takes `--tier`, `--tokens`, and the trainer's intervals, and every one but `tuning`, which has one seed, takes `--seeds`, so the whole program runs in minutes at the `kaggle_t4` tier as a check: `--tier kaggle_t4 --tokens 163840 --warmup-steps 5 --log-every 10 --eval-every 20 --eval-batches 8 --checkpoint-every 10 --no-compiled` is twenty steps per run, with `--seeds 0` where the block takes it and `--recheck-tokens 163840` on `confirmation`, whose re-check otherwise stays at 1B tokens.

A machine with several GPUs runs each command once per GPU, and the workers share the block: each trains the runs no other one holds, claimed by a lock file beside the run's directory (`runs/<block>/.<run>.lock`) that the system frees when its worker dies. Where a block goes on to read what its runs produced (the sweeps before the rates, the calibration runs before the clips, the stress grid before its sixteen-fold level), a worker waits for the runs the others hold before it reads; elsewhere it moves on to the next command of its chain while the others finish. A run gives the same logs bit for bit whichever worker and card of the one model trains it. The blocks go in three chains: the second, RQ3 alone, once Gate 1 is read; the third, on a host of its own, the confirmation tier and then the balancer block, whose runs fill the cards the confirmation's last round leaves idle:

```bash
mkdir -p logs
P="uv run python -m tfg.program"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P tuning --recompute 1 && $P gate1 --recompute 1 && $P rq2 --recompute 1" > logs/first-gpu$gpu.log 2>&1 &
done
```

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P rq3 --recompute 1" > logs/second-gpu$gpu.log 2>&1 &
done
```

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P confirmation --recompute 9 && $P balancer --recompute 1" > logs/third-gpu$gpu.log 2>&1 &
done
```

`tail -f logs/*.log` follows them and `nvidia-smi` shows one process per card. A worker that dies leaves its run to whichever worker next waits for it; a run of a block that does not wait (Gate 1, the RQ3 block, the confirmation's final runs, the balancer block, the sixteen-fold level) is left until the same loop is launched again once every worker has exited, which resumes it from its checkpoint. `evaluate` shares a block the same way, once its training workers have all exited, since a run still held is left out: the same loop with `$P evaluate --block rq3 --batch-size 32` as each worker's command, and `$P evaluate --block confirmation --batch-size 16 && $P evaluate --block balancer --batch-size 32` once the third chain's workers have all exited.

On four cards the first chain took 18.2 hours with the sixteen-fold level (`ai-experiments/chain1_2026_09/`), the second took 14.6 (`ai-experiments/chain2_2026_09/`), and the third takes about 34.6 with the balancer block in the idle cards, 37.5 if one re-check sweep trains a rate past its edge (`ai-experiments/chain3_plan_2026_09/`). The balance-loss block took 5.0 hours of training in one round of four workers, its evaluation 3 minutes (`ai-experiments/balance_loss_2026_09/`). When the first chain's workers have all exited, bring the logs home (last section), read Gate 1 (the analysis above), and delete the saves no later block reads, the tuning runs' and the stress grid's: `find runs/tuning runs/rq2 -name '*.pt' -delete` (the frontier and the stress report read their logs alone; Gate 1's runs keep theirs for the panel and the annex). If the second chain cannot start at once, stop the instance until it can (section 1). When it has ended, evaluate its block, bring everything home, destroy the instance, and run its analysis (section 7) at home, where `runs/tuning` is. A mechanism-tier run's saves are about 4.1 GB (the tuning's 17 came to 69.85 GB); with the tuning's deleted after Gate 1 started, the first chain's host held at most 261 GB, data and environment included. The second chain's saves come to about 135 GB (RQ3's 33 runs), which 200 GB of disk hold with the data and the environment; the third's come to about 185 GB (a confirmation-tier run keeps 6.1 GB under Muon and 7.9 under AdamW, section 8: about 70 for the re-check with an extra rate and 61 for the runs at 3B, and 52 for the balancer block), which 250 GB of disk hold with the data and the environment.

## 6. One run by hand

A block's runs are ordinary runs; one on its own, to redo it or to try something, is:

```bash
uv run python -m tfg.train --tier mechanism --partition muon --seed 3 \
  --steps 15259 --adamw-lr <tuned> --muon-lr <tuned> --compiled \
  --run-dir runs/rq3/mechanism-muon-s3 --resume
```

The arms that depart from default are the same command with one flag more: `--no-balanced` for the balancer-off arms, `--muon-lr-multiplier 0.5` for the arm with Muon at AdamW's rate (`--router-lr-multiplier` also exists, which no block uses). Each flag becomes part of the run's settings and of its default name, and the analysis groups runs by that name, so all the arms can share one directory. `--steps` shapes the cosine schedule and cannot change on resume.

## 7. Analysis

```bash
uv run python -m tfg.analysis --runs-dir runs/rq1 --reference adamw --tuning-dir runs/tuning
uv run python -m tfg.analysis --runs-dir runs/rq3 --reference adamw --tuning-dir runs/tuning
uv run python -m tfg.analysis --runs-dir runs/confirmation --reference adamw --other-tier runs/rq3
uv run python -m tfg.analysis --runs-dir runs/balancer --reference adamw
```

This writes `runs/rq3/analysis/`, and the same for the other blocks. A run directory that has not logged a record yet is left out, and a last line that a crash cut short is not read. The tables: `inventory.csv` (every run: its arm, seed, steps, tokens, final validation loss, and its `status`: `finished`, `unfinished`, or `diverged`, a run that ended on a loss or a gradient that was not a number or that finished at or above its first training loss); `endpoints.csv` (one row per run; every routing reading integrated over the run and, as `_early`, over the 5 percent of its steps that follow the warmup, and, as `_warmup`, over the warmup, the per-mixture ones averaged over the mixtures first and read once more in the first mixture alone with `_first` before the window's suffix (block 1, which the dense block 0 feeds: `bias_span_first`, the primary endpoint, `router_input_similarity_first`, and so on), and the per-group ones, `update_norm_<group>`, `weight_norm_<group>` and `angular_step_<group>` for each parameter group of the partition (`router`, `experts`, `shared`, `attention`, `dense`, `tables_and_norms`, and once more on their own the norm gains, `gains`), as they are; the final validation loss `val_loss` beside the first training loss `loss_initial`; the bias span, usage perplexity, and collapse count at the last logged step as `_final`, and the largest bias span any mixture reached as `bias_span_max`, the collapse count `collapsed_experts` being the experts with under a tenth of their fair share of the batch; plus, from the training loss and the pre-clip norms of every step that each log line carries, the spike counts `spikes` under the plan's definition and `spikes_raw` under Qwen's, both after the warmup and both again per ten thousand steps (`spikes_per_10k`, `spikes_raw_per_10k`, empty for a run that ended inside its warmup), and per optimizer `grad_norm_median_<kind>` after the warmup, `grad_norm_p999_<kind>`, and `clip_fraction_<kind>`); `summary.csv` (per arm); `comparisons.csv` (each arm minus the reference per endpoint, paired by seed over the seeds where both runs finished: the mean difference, its bootstrap interval, the effect size, the difference over the pooled seed spread of the two arms, each seed's sign, and how many agree with the mean's); `contrasts.csv`, the plan's endpoint table: every contrast of every claim of `tfg/claims.py` that the runs' arms allow, the arm predicted `lower` minus the arm predicted `higher`, in each `view` the claim is read in, with the same columns, the seeds both arms were run at (`expected_seeds`), and the `verdict` of the plan's rule on the finished runs: `pass` when every seed's sign is `-`, `direction` when the mean is negative but a seed disagrees, `fail` otherwise, and `incomplete` while a seed both arms have lacks a finished run or the endpoint, or fewer than three have both; and `claims.csv`, one row per claim, its `confirmatory` column telling the plan's registered claims from the exploratory C7, with its verdict in each view and overall, which the report also prints: a claim passes when every one of its contrasts does, is a `fail` once one of them failed, a `direction` when none failed but one is a `direction`, and `incomplete` otherwise. A verdict is final once the block has finished: the analysis knows the seeds the runs have, not the ones still to be launched, so a contrast can read `pass` on a block's first seeds while its last ones have not started. The comparisons come twice, the `matching` column tells them apart: at equal `tokens`, and at matched `loss`, where, pair of arms by pair of arms and seed by seed, both runs are cut at the evaluation where each first reaches the worse of their two final validation losses (`matched_step`; the worse run keeps its last step unless its validation loss had been that low before), with those cut endpoints in `endpoints_matched.csv`, the pair in its `first` and `second` columns, the two arms in name order, which says nothing of which one a claim predicts lower. The analysis refuses a directory whose runs did not all train on the same number of windows (`train_windows`), since then a seed did not give them the same data order. The plan's one comparison that is not against (a), (c) against (b) on the validation loss beside C5, is the `muon_except_router` row of `val_loss` in the `comparisons.csv` of a second pass, `--reference muon --out-dir runs/rq3/analysis-muon`.

The figures are the plan's list, each written as a PNG to look at and as a PDF of the same name for the thesis. One per curve with a line per arm and the band its seeds span (`bias_span.png`, `load_churn.png`, and so on), the no-balancer arms' usage perplexity on its own (`usage_perplexity_unbalanced.png`), and the per-block view of every mixture reading (`<reading>_blocks.png`). The early window, churn and router entropy through the warmup and the 5 percent after it for the router-position arms (a), (b), (c), (g) (`early_window.png`). The C6 panel: `update_load_scatter.png`, each expert's update Frobenius norm against its share of the batch at three points in training, a row per arm (its lowest seed), and `_spectral`, the same with the spectral norm, for the appendix. The strips, one per confirmatory endpoint, per arm with a point per seed (`<endpoint>_seeds.png`: `bias_span_first_seeds.png` for the primary endpoint, `router_input_similarity_first_seeds.png`, `load_churn_warmup_seeds.png`, `update_load_correlation_seeds.png`, and so on). The geometry figure (`geometry.png`): per run, the first mixture's integrated router-input similarity against its integrated bias span, a color per arm. The exploratory grid (`exploratory.png`): margins, logit norm, SVD entropy, row similarity, the expert update dispersion, kurtosis, collapse, the stability maxima, and every group's weight norm and angular step, with Kosson et al.'s equilibrium angle for the all-AdamW arm dashed on the decayed groups' angular steps. The primary endpoint in its two views side by side (`views.png`). With `--other-tier`, the confirmation figure (`tiers.png`): the confirmatory endpoints the runs have per arm, a point per seed, a marker per tier. With `--tuning-dir`, the frontier (`frontier.png`, its table in `frontier.csv`): for each of the tuning block's three sweeps, AdamW's, Muon's and the momentum control's, the final validation loss of every finished rate against its integrated bias span, over every mixture in one panel and in the first mixture alone in the other, the points joined in rate order, labeled by the rate, the rate that won filled in.

When the runs carry the panel (section 8), the analysis also draws the specialization figure: `specialization_heatmap.png`, the share of each domain's tokens per expert in the last mixture against a uniform share, one seed per arm, and `specialization_seeds.png`, every seed's divergence with the arm's mean. When they carry the annex readings too, the per-arm summary of those alone goes to `annex.csv`, the appendix's table, and each reading gets its per-mixture plot.

## 8. Specialization panel

Four domain slices of 2M tokens each: the web one is cut from a held-out FineWeb-Edu validation shard, the other three (`open-web-math`, `codeparrot-clean-valid`, Wikipedia `20231101.en`) stream from the Hub and are tokenized with the thesis tokenizer, rebuilt with `tiktoken` from its merge list and checked identical to the encoder that made the shards on 300 texts, 75 from each domain (`ai-experiments/platform_checks_2026_09/tokenizer_identity.py`):

```bash
uv run python -m tfg.panel build
```

This writes `data/panel/` (the four `.npy` slices and the whitespace mask), a few minutes on the network. Then, per finished run:

```bash
uv run python -m tfg.panel evaluate --run-dir runs/rq3/<run>
```

sends every slice through the run's checkpoint and writes `specialization.json` in the run directory: per mixture, the share of each domain's tokens that reaches each expert (summing to K, against a uniform K / E) and the Jensen-Shannon divergence between the four domains' usage distributions, in nats, zero when every domain uses the experts alike. The code slice is counted again without its whitespace tokens (`specialization_nonspace`). The analysis picks the file up as the `specialization` endpoints, averaged over the mixtures, and plots them per mixture.

The annex readings run over the same panel:

```bash
uv run python -m tfg.annex --run-dir runs/rq3/<run> --batch-size 32
```

(16 at the confirmation tier, where 32 does not fit either: section 4) writes `annex.json` in the run directory, per mixture: router saturation (the share of the experts the end of the run chooses that were already chosen one percent into it, read from the model-only `early.pt` the trainer keeps at that point), expert co-activation (how often each pair of experts is chosen together, normalized by the first's count), routing purity (how concentrated each expert's tokens are on one domain), the hidden-state cosine between and within domains beside the routing cosine, the cosine between two experts' outputs on the same tokens, and the alignment between each router row's gradient and its expert's input-weight gradient, from one backward pass over the first batch of each domain. The chance levels of co-activation, routing purity, and the gradient alignment go in the file too. The analysis averages each reading over the mixtures and summarizes them per arm in `annex.csv`. The pass takes about twice the panel's evaluation (40 against 20 seconds at the T4 tier with 100k tokens per domain); on a GPU the counted readings move in the sixth decimal with `--batch-size`, since the forward's rounding depends on the shape and a few near-tie tokens change expert. Two readings move with it outright, the expert similarity and the gradient alignment, which are taken on the first batch of each domain: evaluate every run of a block with the same `--batch-size`, which `annex.json` records as `batch_size`, and compare those two readings only between runs that share it. `early.pt` is the model alone, a third of the checkpoint's size under AdamW and about half on the Muon partitions, whose matrices keep one moment instead of two. A mechanism-tier run keeps 3.7 GB of the two files under Muon and 4.9 under AdamW, and a confirmation-tier run 6.1 and 7.9, about 580 GB over the whole program with every extra run, more than the disk holds: the tuning and stress runs' go between the chains (section 5), and a finished run's `checkpoint.pt` and `early.pt` can go once `evaluate` has written its panel and annex files, since a relaunched block leaves such a run alone.

## 9. Stress runs (RQ2)

Each partition is trained at 1x, 2x, 4x, and 8x its tuned rates on a constant schedule after the warmup, three seeds each, 0.5B tokens, and at 16x when none of its runs at 8x diverged. `--lr-multiplier` scales both tuned rates and joins the run's name as `-lr<m>x`. Clipping is per optimizer, at twice the post-warmup median pre-clip norm of that optimizer in the unstressed run. The `rq2` block (section 5) does this: it trains the calibration run at 1x under `runs/rq2/calibration/`, out of the stress analysis's way, reads `grad_norm_median_adamw` and `grad_norm_median_muon` from its endpoints (section 7), and passes the doubles to every arm of the family, the 1x one included, so the whole family clips alike; `momentum` shares Muon's threshold. The balancer's update rate stays at its default in every primary arm; the one control arm scales it with the multiplier, `--balancer-multiplier 4` on a 4x run, and is a family of its own in the stress analysis (`muon-lr4x-balancer4x` belongs to `muon-balancer4x`). One stressed run by hand:

```bash
uv run python -m tfg.train --tier mechanism --partition muon --steps 7629 --compiled \
  --adamw-lr <tuned> --muon-lr <tuned> --schedule constant --lr-multiplier 4 \
  --adamw-clip <2x median> --muon-clip <2x median> \
  --seed 0 --run-dir runs/rq2/mechanism-muon-lr4x-s0 --resume
```

Every log line carries the training loss and each optimizer's pre-clip norm of every step, so the stress analysis needs nothing beyond the run directories:

```bash
uv run python -m tfg.stress --runs-dir runs/rq2
```

This writes `runs/rq2/stress/`: `stress.csv` (one row per run: the endpoints of section 7 with the run's `family`, its arm at 1x, `loss_capped`, the final validation loss capped at the initial training loss so a diverged run counts as its start, empty for a run still under way, and the `_peak` of each activation reading, the largest value logged in the run), `summary.csv` (mean, spread, and count over seeds per family and multiplier), `sensitivity.csv` (Wortsman's rate sensitivity per family: the mean over the multipliers 1x to 8x of how far the capped loss, averaged over seeds, sits above the family's best; a family has one once a run of it has ended at every one of those multipliers, so the control family, run at 4x alone, has a point on the curve and no number, and neither has any family of a block that runs part of the grid, as in the optional confirmation-tier subset), the four curves against the multiplier (`sensitivity`, `spikes` per ten thousand steps under both definitions, `activations`, and `churn`, the load churn in the warmup window, early, and over the run, each as a PNG and a PDF), and the event bank: `events.csv` lists every primary-spike episode (consecutive spike steps merged) with its peak step, loss, excess over the rolling median, and whether the excess is `large`, above 0.5; `traces.csv` holds the routing readings of every mixture, with the step's loss and gradient norm, logged within 200 steps of each peak, aligned by their `offset` from it (the loss and the norm are the logged steps' own, so a spike between two logged steps shows in `events.csv`, which is read from every step, and not here), `traces.png` draws them so aligned, a line per episode colored by family, and `onsets.csv` says, per episode and traced reading, the offset from the peak at which the reading first rose above its baseline and was still above it at the next logged step (its median over the offsets from -200 to -100, plus five robust standard deviations of it there; empty when it never did, when the run has no baseline that far before the peak, or when the baseline does not vary), which is the order of events H4 asks about. The rows with `placebo` set are the control of that reading: the same onsets around the step 400 before each peak (their `peak` column is that step), where nothing happened, kept only when the step falls after the warmup, as every episode does, and no episode of the run touches that window. A reading that has onsets there as well rises by drifting, and its onsets before a spike say nothing of the spike.

## Before the rented cards: a Kaggle T4

Every entry point through section 9 was run on Kaggle's free T4 on 2026-09-12 (tests, every partition, the flag-named arms and the β2 check, the same run twice, `--compiled` against eager, resume, a sweep, the panel build and evaluation, the analysis with its windows, per-block figures and matched-loss view, the stress runs at 1x and 4x with their analysis, memory at the mechanism and confirmation tiers, and the telemetry's share of a step), and the annex the same day (four 200-step runs, one compiled, each keeping `early.pt` at step 2; the annex on every run, twice on one at two batch sizes; every reading in range and above chance; the analysis with `annex.csv` and the specialization figure). The program (section 5) ran there on 2026-09-13 at twenty steps per run: the tuning block with two AdamW rates and one Muon rate and the β2 check, Gate 1 compiled, the factorial at one seed cut after 110 seconds and relaunched (five runs found finished, the rest trained), the stress grid at 1x and 4x with its three calibration runs, the confirmation re-check and its three runs at forty steps, the panel and the annex over the factorial's twelve runs, once more with every run skipped, and the three analyses; every run's settings, clips, and rates checked afterwards against what the plan fixes. The whole pass, tests included, took forty minutes; a twenty-step run is about half a minute there, its evaluation another half. The pass ran again on the finished code on 2026-09-18, in 61 minutes, with what it lacked: a run at a million times the rate, which diverges on step 2 and which a relaunch leaves alone, the stress grid at 1x to 8x with that multiplier added so that its analysis reads diverged runs, a second stress block whose 16x extension trains, the confirmation block under `--recompute` with its annex at a smaller batch, a run whose checkpoint stands behind its whole log (skipped by the evaluation, finished by a relaunch, then evaluated, and trained again without `--resume` in an emptied directory), and the recompute readings of section 4. The kernel and its outputs are in `ai-experiments/platform_checks_2026_09/kaggle_e2e/`. The repository is private, so the code goes up as a private Kaggle dataset and a script kernel installs it, all from the shell:

```bash
COPYFILE_DISABLE=1 tar czf <dataset dir>/code.tar.gz --exclude __pycache__ pyproject.toml uv.lock RUNBOOK.md tfg
kaggle datasets version -p <dataset dir> -m "<what changed>"   # `datasets create` the first time
kaggle kernels push -p <kernel dir> --accelerator NvidiaTeslaT4
kaggle kernels status marcoshernanz/tfg-e2e-t4
```

Kaggle unpacks the tarball on upload, so the kernel copies the folder holding `pyproject.toml` to `/tmp/code` (anything under `/kaggle/working` is saved as output, including a venv), runs `pip install uv && uv sync --frozen` (half a minute; the box's CUDA 13 driver takes the locked torch), downloads three train shards, and runs `uv run python -m tfg.train --tier kaggle_t4 ...`. Give every step of the kernel a time limit (start it in its own session, wait with a timeout, kill the process group on expiry) and `set -o pipefail`: the CLI cannot cancel a running kernel, so one step that never exits holds the GPU until Kaggle's own cap. The panel build was that step once, a Hub parquet stream with a read still in flight at interpreter exit, which `tfg.panel` waits out before it returns.

What the T4 gives, and does not: `--tier kaggle_t4` trains at about 7k tokens/s with Muon and 10k with AdamW; `--compiled` is correct there (loss within 5e-5 of eager) but no faster, since the compiler skips bf16 on that generation; the mechanism tier fits only with `--batch-size 8` (peak 11.9 of 14.6 GiB, 16 runs out of memory) at about 2.3k tokens/s, so one 0.5B-token sweep run would take two and a half days there; the confirmation tier fits at `--batch-size 4` (13.6 GiB) and 2 (10.2 GiB). The telemetry adds 0.58 s to a 1.2 s `kaggle_t4` step and 1.4 s to a 3.9 s mechanism step at batch 8, about 4 to 5 percent of training at `--log-every 10`; the attention-logit recomputation alone is 32 and 95 ms of that. It checks the code; it does not run the plan.

## Bringing results home

Sync the run directories without their checkpoints and early saves, gigabytes each (the panel and the annex, which need them, run on the host first: `tfg.program evaluate`), every few hours while a chain runs and once more before the instance is stopped or destroyed:

```bash
rsync -az --exclude '*.pt' --exclude '.*.lock' -e "ssh -p <port>" root@<ip>:code/runs/ ~/dev/tfg/code/runs/
```

Then copy the program's blocks to the private dataset `marcoshernanz/tfg-runs` on the Hub, which keeps every version (the first chain's went up on 2026-09-24, the second's on 2026-09-26, the third's on 2026-09-28), and add each new folder to the dataset's card:

```bash
hf upload marcoshernanz/tfg-runs runs . --repo-type dataset --include "tuning/**" --include "rq1/**" --include "rq2/**" --include "pilot/**" --include "rq3/**" --include "confirmation/**" --include "balancer/**" --include "balance_loss/*-balanceloss*/**" --include "balance_loss/analysis/**"
```

(the balance-loss block's own runs and analysis alone: its links to RQ3's runs would upload those again)

and bring them back into `code/runs/` on any machine with `hf download marcoshernanz/tfg-runs --repo-type dataset --local-dir runs --exclude README.md --exclude "offset-probe/**" --exclude "scale-probe/**"`. The dataset also keeps the two probes' runs, which belong under `ai-experiments/`, not in `runs/`: the offset probe's under `offset-probe/` (2026-09-26), from `ai-experiments/rq3_reading_2026_09/offset_probe/outputs/mechanism/runs/`, and the scale probe's under `scale-probe/` (2026-09-28), from `ai-experiments/scale_probe_2026_09/outputs/runs/`.
