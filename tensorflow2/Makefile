#
# Author: Jake Zimmerman <jake@zimmerman.io>
#
# ===== Usage ================================================================
#
# NOTE:
#   When running these commands at the command line, replace $(TARGET) with
#   the actual value of the TARGET variable.
#
#
# make                  Compile all *.md files to PDFs
# make <filename>.pdf   Compile <filename>.md to a PDF
# make <filename>.tex   Generate the intermediate LaTeX for <filename>.md
#
# make view             Compile $(TARGET).md to a PDF, then view it
# make again            Force everything to recompile
#
# make clean            Get rid of all intermediate generated files
# make veryclean        Get rid of ALL generated files:
#
# make print            Send $(TARGET).pdf to the default printer:
#
# ============================================================================

TARGET=sample

all: $(patsubst %.md,%.pdf,$(wildcard *.md))

PANDOC_FLAGS =\
	-f markdown+tex_math_single_backslash \
	-t latex \

LATEX_FLAGS = \

PANDOC_FLAGS += --variable mainfont="SimSun" \
	--variable sansfont="SimSun" \
	--variable monofont="SimSun" \
	--variable fontsize=12pt \
	--from markdown \
	--template eisvogel \
	-V CJKmainfont=SimSun

PDF_ENGINE = xelatex

PANDOCVERSIONGTEQ2 := $(shell expr `pandoc --version | grep ^pandoc | sed 's/^.* //g' | cut -f1 -d.` \>= 2)
ifeq "$(PANDOCVERSIONGTEQ2)" "1"
    LATEX_FLAGS += --pdf-engine=$(PDF_ENGINE)
else
    LATEX_FLAGS += --latex-engine=$(PDF_ENGINE)
endif

all: $(patsubst %.md,%.pdf,$(wildcard *.md))

# Generalized rule: how to build a .pdf from each .md
%.pdf: %.md
	pandoc $(PANDOC_FLAGS) $(LATEX_FLAGS) -o $@ $<

# Generalized rule: how to build a .tex from each .md
%.tex: %.md
	pandoc --standalone $(PANDOC_FLAGS) -o $@ $<

touch:
	touch *.md

again: touch all

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb || true

veryclean: clean
	rm -f *.pdf

view: $(TARGET).pdf
	if [ "Darwin" = "$(shell uname)" ]; then open $(TARGET).pdf ; else xdg-open $(TARGET).pdf ; fi

print: $(TARGET).pdf
	lpr $(TARGET).pdf

.PHONY: all again touch clean veryclean view print
