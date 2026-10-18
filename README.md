# How the Optimizer Shapes Routing

The code and the LaTeX source of the bachelor's thesis *How the Optimizer Shapes Routing: Load Balancing, Specialization, and Stability in Mixture-of-Experts Language Models Trained with Muon*, by Marcos Hernanz Antón, supervised by Daniel Hernández-Lobato (Grado en Ingeniería Informática, Escuela Politécnica Superior, Universidad Autónoma de Madrid).

## Contents

- `code/`: the Python package `tfg`, which trains and analyzes every run of the thesis, with its tests in `code/tfg/tests/`, its `pyproject.toml`, and its lockfile `uv.lock`. `code/RUNBOOK.md` gives the full procedure on a rented host.
- `code/ai-experiments/`: the scripts that make the figures and tables of the Results (`thesis_figures_2026_09/`), the scripts of the probes and of the exploratory analyses, and the scripts that ran each chain of runs, frozen with their outputs. The package `tfg` imports nothing from them.
- `thesis/tex/`: the memoria. It is built in the EPS class `tfgtfmthesisuam`, which is in `thesis/tex/tfgtfmthesisuam/` exactly as the school distributes it, under its own license, the AGPL-3.0.

Some files refer to `context/`, for example to `context/decision/EVIDENCE-LOG.md`. These are the working notes of the thesis, which are not published.

## Data

- The training data, tokenized: [`marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full`](https://huggingface.co/datasets/marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full).
- The runs: [`marcoshernanz/tfg-runs`](https://huggingface.co/datasets/marcoshernanz/tfg-runs) holds the directory of every run of the thesis and of the two probes.

Both datasets are public and need no account.

## Running the code

```bash
cd code
uv sync --frozen
uv run pytest -q
```

Appendix C of the thesis lists the commands that train and analyze every run.

## Building the thesis

```bash
cd thesis/tex
latexmk
```

The build uses pdfLaTeX and writes the PDF to `thesis/tex/build/main.pdf`.
