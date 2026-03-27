PAPER = main

default:
	pdflatex $(PAPER).tex
	bibtex --min-crossrefs=99 $(PAPER)
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex
	rm -f *.aux *.bbl *.blg *.gz *.fls *.log *.nav *.out *.snm *.toc *.vrb
	rm -f */*.aux */*.log *.fdb_latexmk

noc:
	pdflatex "\def\nocflagdefined{}\input{$(PAPER).tex}" $(PAPER).tex
	bibtex --min-crossrefs=99 $(PAPER)
	pdflatex "\def\nocflagdefined{}\input{$(PAPER).tex}" $(PAPER).tex
	pdflatex "\def\nocflagdefined{}\input{$(PAPER).tex}" $(PAPER).tex

clean:
	rm -f *.aux *.bbl *.blg *.gz *.fls *.log *.nav *.out *.snm *.toc *.vrb
	rm -f */*.aux */*.log 
