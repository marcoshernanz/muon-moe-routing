# Muon Makes the First Mixture-of-Experts Layer Harder to Balance

The code and the LaTeX source of the paper [*Muon Makes the First Mixture-of-Experts Layer Harder to Balance*](https://arxiv.org/abs/2610.37914), by Marcos Hernanz, arXiv:2610.37914 (October 2026). Every run is public on Hugging Face (see [Data](#data)).

In mixture-of-experts language models with 55M and 108M active parameters, Muon on the rest of the network makes the inputs to the router of the first MoE layer share a direction, and soon after the warmup the busiest expert of that layer averages four to seven times its fair share. Moving only the router to AdamW made the loads shift more during the warmup, while subtracting the batch mean from the router's inputs removed the overload over the first 3,000 steps.

The paper is the short version of the bachelor's thesis *How the Optimizer Shapes Routing: Load Balancing, Specialization, and Stability in Mixture-of-Experts Language Models Trained with Muon*, by Marcos Hernanz Antón, supervised by Daniel Hernández Lobato (Grado en Ingeniería Informática, Escuela Politécnica Superior, Universidad Autónoma de Madrid), whose LaTeX source is also here.

## Contents

- `code/`: the Python package `tfg`, which trains and analyzes every run of the thesis, with its tests in `code/tfg/tests/`, its `pyproject.toml`, and its lockfile `uv.lock`. `code/RUNBOOK.md` gives the full procedure on a rented host.
- `code/ai-experiments/`: the scripts that make the figures and tables of the Results (`thesis_figures_2026_09/`) and of the paper (`paper_figures_2026_10/`), the scripts of the probes and of the exploratory analyses, and the scripts that ran each chain of runs, frozen with their outputs. The package `tfg` imports nothing from them.
- `paper/`: the LaTeX source of the short paper, [*Muon Makes the First Mixture-of-Experts Layer Harder to Balance*](https://arxiv.org/abs/2610.37914), built with `latexmk` into `paper/build/main.pdf`.
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

## Citation

```bibtex
@misc{hernanz2026muon,
  title         = {Muon Makes the First Mixture-of-Experts Layer Harder to Balance},
  author        = {Hernanz, Marcos},
  year          = {2026},
  eprint        = {2610.37914},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.37914}
}
```

## License

Everything under `code/` is under the MIT license (`LICENSE`). The memoria in `thesis/tex/` and the paper in `paper/` are not: the memoria's rights page and the paper's arXiv license apply to them. The template in `thesis/tex/tfgtfmthesisuam/` keeps its own license, the AGPL-3.0.
