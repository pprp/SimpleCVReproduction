---
documentclass: homework
# For option documentation, see https://github.com/jez/latex-homework-class
classoption:
- 11pt
- largemargins
# Remove or change this line if you don't have the Menlo font installed
monofont: Menlo
newtxmathoptions:
- cmintegrals
- cmbraces
colorlinks: true
linkcolor: RoyalBlue
urlcolor: RoyalBlue
header-includes:
- '\newcommand{\hwname}{Jacob Zimmerman}'
- '\newcommand{\hwemail}{jezimmer}'
- '\newcommand{\hwtype}{Homework}'
- '\newcommand{\hwnum}{1}'
- '\newcommand{\hwclass}{15-123}'
- '\newcommand{\hwlecture}{1}'
- '\newcommand{\hwsection}{A}'
---

\maketitle

#

_Count the rectangles of all sizes and of all positions that are formed using
segments in a grid with \(m\) horizontal and \(n\) vertical lines._

_Solution._ A rectangle is uniquely described by two distinct vertical lines and
two distinct horizontal lines. Therefore we can just select these two lines and 
multiply these quantities by rule of product:

<!-- LaTeX environment in a Markdown document! -->
\begin{align*}
  \binom{n}{2}\binom{m}{2}
\end{align*}


# Anagrams {-}

_How many ways are there to rearrange the letters in the word "anagram"?_

_Solution._ We can choose an arrangement of the letters in "anagram" in two
steps. We first choose 3 of the 7 positions to be a's, then permute "ngrm" in
the remaining positions. Thus, we have
\begin{align*}
  \binom{7}{3} 4!
\end{align*}

ways to choose an arrangement.


董佩杰