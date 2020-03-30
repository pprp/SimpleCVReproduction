# tufte-handout

These starter files can be used for writing up articles formatted in a style
formatted by Edward Tufte. It uses the [`tufte-handout`] document class. See
that project for more usage information. Alternatively, take a look at a [sample
document] using the `tufte-handout` class.

[`tufte-handout`]: https://www.ctan.org/pkg/tufte-latex?lang=en
[sample document]: http://ctan.sharelatex.com/tex-archive/macros/latex/contrib/tufte-latex/sample-handout.pdf

## Dependencies

There are a number of build dependencies for these starter files:

- [Pandoc], a universal document converter
- [LaTeX], a document preparation system
- _Optional_: The font Menlo
- _Optional_: The font Palatino

[Pandoc]: http://pandoc.org/
[LaTeX]: https://www.latex-project.org/

Installation instructions vary depending on your system. See the linked websites
for more information.

## Usage

1. Copy these starter files wherever you'd like.
1. Rename `sample.md` to `<filename>.md`
1. Edit the `TARGET` variable at the top of the [Makefile] to equal `<filename>`
1. Write your content in `<filename>.md`
    - Be sure to adjust the information like the `title` and `author` at the top
      of the file
1. Read the [Makefile's documentation][Makefile].

[Makefile]: src/Makefile

## License

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://jez.io/MIT-LICENSE.txt)
