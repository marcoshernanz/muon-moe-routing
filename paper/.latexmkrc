# pdfLaTeX, BibTeX when the citations change, every generated file under build/
$pdf_mode = 1;  # 1 is pdflatex (what arXiv and Overleaf run by default); 4 is lualatex
$out_dir = 'build';
$bibtex_use = 1;  # run bibtex as needed, and keep the .bbl on `latexmk -c` (arXiv needs it)
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
@default_files = ('main.tex');
