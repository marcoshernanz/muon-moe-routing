# The data of the memoria's Results (2026-09-29)

Scripts the assistant wrote to read the finished runs in `code/runs/` into the figures, tables and numbers of the memoria's Results chapter (`thesis/tex/sections/results.tex`). They train nothing and change no run. The endpoints and the paired comparisons are those of `tfg.analysis` and `tfg.claims`; the curves are read from the routing logs with `tfg.analysis`'s readers.

Run from `code/`.

## Scripts

| Script | What it does | Output |
|---|---|---|
| `extract.py` | One CSV per data figure (RQ1's span by layer and over training, the tuning curves, the stress runs, RQ3's geometry and its first 4,000 steps), and every number the chapter's text quotes, with its paired interval and signs. | `thesis/tex/figures/data/*.csv`, `outputs/numbers.txt` |
| `tables.py` | The seed means and ranges of every reading the chapter's tables use, and the paired differences with `tfg.claims.compare`'s interval, block by block (RQ3, the bias speed, the balance loss, the large model). | `outputs/tables.txt` (redirected) |

```bash
.venv/bin/python ai-experiments/thesis_figures_2026_09/extract.py
.venv/bin/python ai-experiments/thesis_figures_2026_09/tables.py > ai-experiments/thesis_figures_2026_09/outputs/tables.txt
```

The figures that read the CSVs are `thesis/tex/figures/results-*.tex`, drawn by pgfplots with the styles `thesis` and `band` of `thesis/tex/main.tex`. A number in the chapter that is in neither output comes from `context/decision/EVIDENCE-LOG.md` (sections 17 to 23), which the chapter's comments name.
