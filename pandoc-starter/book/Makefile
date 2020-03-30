####################################################################################################
# Configuration
####################################################################################################

# Build configuration

BUILD = build
MAKEFILE = Makefile
OUTPUT_FILENAME = book
METADATA = metadata.yml
CHAPTERS = chapters/*.md
TOC = --toc --toc-depth=2
METADATA_ARGS = --metadata-file=$(METADATA)
IMAGES_FOLDER = images
IMAGES = $(IMAGES_FOLDER)/*
COVER_IMAGE = $(IMAGES_FOLDER)/cover.png
MATH_FORMULAS = --webtex
CSS_FILE = style.css
CSS_ARGS = --css=$(CSS_FILE)

# Debugging

# DEBUG_ARGS = --verbose

# Pandoc filtes - uncomment the following variable to enable cross references filter. For more
# information, check the "Cross references" section on the README.md file.

# FILTER_ARGS = --filter pandoc-crossref

# Combined arguments

ARGS = $(TOC) $(MATH_FORMULAS) $(CSS_ARGS) $(METADATA_ARGS) $(FILTER_ARGS) $(DEBUG_ARGS)
PANDOC_COMMAND = pandoc

# Per-format options

EPUB_ARGS = --epub-cover-image=$(COVER_IMAGE)
HTML_ARGS = --standalone --to=html5
PDF_ARGS = --pdf-engine=xelatex -V geometry:margin=1in -V documentclass=report \
		   -V CJKmainfont="Microsoft YaHei" \
		   -M mainfont:"SimSun" -M sansfont:"SimSun" -M  monofont:"SimSun" \
		   --template pm-template \
		   --listings

#  -M  mathfont:"SimSun"
#-V CJKmainfont="HiraginoSans-W4" \
#-V mainfont="SimSun" 
#-V CJKmainfont=IPAexGothic
DOCX_ARGS =

####################################################################################################
# Basic actions
####################################################################################################

all:	book

book:	pdf
#epub html pdf docx

clean:
	rm -r $(BUILD)

####################################################################################################
# File builders
####################################################################################################

# epub:	$(BUILD)/epub/$(OUTPUT_FILENAME).epub

# html:	$(BUILD)/html/$(OUTPUT_FILENAME).html

pdf:	$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf

# docx:	$(BUILD)/docx/$(OUTPUT_FILENAME).docx

# $(BUILD)/epub/$(OUTPUT_FILENAME).epub:	$(MAKEFILE) $(METADATA) $(CHAPTERS) $(CSS_FILE) $(IMAGES) \
# 		$(COVER_IMAGE)
# 	mkdir -p $(BUILD)/epub
# 	$(PANDOC_COMMAND) $(ARGS) $(EPUB_ARGS) -o $@ $(CHAPTERS)
# 	@echo "$@ was built"

# $(BUILD)/html/$(OUTPUT_FILENAME).html:	$(MAKEFILE) $(METADATA) $(CHAPTERS) $(CSS_FILE) $(IMAGES)
# 	mkdir -p $(BUILD)/html
# 	$(PANDOC_COMMAND) $(ARGS) $(HTML_ARGS) -o $@ $(CHAPTERS)
# 	cp -R $(IMAGES_FOLDER)/ $(BUILD)/html/$(IMAGES_FOLDER)/
# 	cp $(CSS_FILE) $(BUILD)/html/$(CSS_FILE)
# 	@echo "$@ was built"

# $(BUILD)/docx/$(OUTPUT_FILENAME).docx:	$(MAKEFILE) $(METADATA) $(CHAPTERS) $(CSS_FILE) $(IMAGES)
# 	mkdir -p $(BUILD)/docx
# 	$(PANDOC_COMMAND) $(ARGS) $(DOCX_ARGS) -o $@ $(CHAPTERS)
# 	@echo "$@ was built"

$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf: $(MAKEFILE) $(METADATA) $(CHAPTERS) $(CSS_FILE) $(IMAGES) 
	mkdir -p $(BUILD)/pdf 
	$(PANDOC_COMMAND) $(ARGS) $(PDF_ARGS) -o $@ $(CHAPTERS)
	@echo "$@ was built"

