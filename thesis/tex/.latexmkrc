# One build for the command line and for VS Code: pdfLaTeX, BibTeX when the citations change, every generated file under build/. The class and the logos of the cover are read from tfgtfmthesisuam/, which holds the EPS template as distributed.
$pdf_mode = 1;  # 1 is pdflatex, which the class is written for
$out_dir = 'build';
$bibtex_use = 2;  # run bibtex as needed, and let `latexmk -c` remove the .bbl
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
ensure_path('TEXINPUTS', './tfgtfmthesisuam');
@default_files = ('main.tex');
