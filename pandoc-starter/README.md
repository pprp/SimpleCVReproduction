# Pandoc Starter

This repository contains a number of files for getting started with Pandoc. Each
example comes with:

- A `README.md`, describing the type of document and usage
- A `Makefile`, for building the document
- A preview of the document
- The relevant starter files (usually Markdown, sometimes LaTeX)

pandoc-latex-book: 适合将所有的md文件合称为一个以后，然后再运行：

```shell
pandoc ssd.md -o ssd.pdf --from markdown --template eisvogel --pdf-engine=xelatex -V CJKmainfont=SimSun --toc
```

markdown2pdf

```shell
pandoc -s --toc --pdf-engine=xelatex  -o faster.pdf   metadata.yaml --template eisvogel 
```

