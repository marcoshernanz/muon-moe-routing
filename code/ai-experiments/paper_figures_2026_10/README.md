# The data of the arXiv paper (2026-10-24)

Scripts the assistant wrote to read the finished runs into the figures, tables and numbers of the short arXiv version of the thesis (`paper/main.tex`). They train nothing and change no run. The paper reuses the memoria's figure data where it can (`../thesis_figures_2026_09/`) and these scripts add only what it needs on top.

Run from `code/`.

## Scripts

| Script | What it does | Output |
|---|---|---|
| `extract.py` | The one figure data file the memoria does not have: the first MoE layer over the first 3,000 steps of the offset probe's runs with the first router's input centered (seeds 3 to 5), binned as in the memoria, with a check that the probe's plain runs fall inside their arm's band. | `paper/figures/data/rq3-early-muon-centered.csv` |
| `paper_numbers.py` | Every number of the paper that comes from a block's endpoints, recomputed at full precision from `runs/<block>/analysis/endpoints.csv` on the public Hub dataset `marcoshernanz/tfg-runs` (cached in `outputs/hub/`): the mean and range of every arm and every paired difference with its range and signs. | `outputs/numbers.txt` (redirected) |
| `tables.py` | The rows of the paper's tables (the main comparison, the ablation, the rise across experiments and the optimizer of the router) as LaTeX, each value rounded once from full precision. | `outputs/tables.txt` (redirected) |
| `compute.py` | The GPU-hours of the training steps of every run, from the `tokens_per_s` of each logged record of every `metrics.jsonl` on the Hub dataset (cached in `outputs/hub/`), by block and for the 145 runs of the program, without the pilot and the two probes. Appendix B of the paper reports the program's 252.6. | `outputs/compute.txt` (redirected) |

```bash
.venv/bin/python ai-experiments/paper_figures_2026_10/extract.py
.venv/bin/python ai-experiments/paper_figures_2026_10/paper_numbers.py > ai-experiments/paper_figures_2026_10/outputs/numbers.txt
.venv/bin/python ai-experiments/paper_figures_2026_10/tables.py > ai-experiments/paper_figures_2026_10/outputs/tables.txt
.venv/bin/python ai-experiments/paper_figures_2026_10/compute.py > ai-experiments/paper_figures_2026_10/outputs/compute.txt
```

The figures are drawn by pgfplots from `paper/figures/src/*.tex` (`paper/figures/src/build.sh`), reading the CSV files in `paper/figures/data/`, which are this folder's file and copies of the memoria's. A number in the paper that is in none of the outputs comes from `../thesis_figures_2026_09/outputs/numbers.txt` or from `context/decision/EVIDENCE-LOG.md` (sections 19 and 21 for the probes), as the comment at the top of `paper/main.tex` says.
