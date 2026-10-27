#!/bin/sh
# Builds every figure of the paper from its standalone source in this folder and copies the PDF to ../, where main.tex reads it.
set -e
cd "$(dirname "$0")"
for figure in span arms early geometry stress; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$figure.tex" >/dev/null
  cp "build/$figure.pdf" ..
done
