# 从markdown到pdf
Cheng Jun

本模板适合学习笔记等场景：
- 用markdown写作，便捷
- 然后汇编成pdf，便于打印和阅读

模板文件在**markdown2pdf**中。
```PowerShell
PS C:\Users\cheng\markdown2pdf> tree /f
C:.
│  README.md
│
└─markdown2pdf
    │  01-引言.md
    │  02-Julia编程基础.md
    │  03-统计学基础.md
    │  book.ps1
    │  Julia与数据分析.pdf
    │  metadata.yaml
    │  template.tex
    │
    └─image
            JuliaLogo.png
            ps.jpg
            vscode.jpg
            yaml.jpg
```

**Julia与数据分析.pdf**是汇总生成的pdf文件。

## md是markdown文件，是笔记的内容  
这个是用户自行撰写，按照“**-标题”的格式写，本设计是一章节一个md文件
```text
│  01-引言.md
│  02-Julia编程基础.md
│  03-统计学基础.md
```

## book.ps1是powershell脚本，主要用来自动化编译LaTeX  
```PowerShell
$i = dir -name  -filter *md
pandoc -N -s --toc --pdf-engine=xelatex  -o Julia与数据分析.pdf   metadata.yaml --template=template.tex $i 
```
可以修改文件名，此时的演示文件名是“Julia与数据分析.pdf”
- metdata.yaml是标注pdf属性的文件
```yaml
---
title:  'Julia与数据分析'
author:
- 实证小青年
...
```

pdf属性显示如下：

![pdf属性](markdown2pdf/image/yaml.jpg)

## template.tex是模板  
template.tex模板是从R语言包[rticles](https://cran.rstudio.com/web/packages/rticles/)中default.latex文件修改而来。[rticles](https://cran.rstudio.com/web/packages/rticles/)中的这个文件是从[pandoc](http://www.pandoc.org)中的默认模板修改而来。

例如字体设置部分如下：
```LaTeX
% 字体的设置，可以自行修改
% \usepackage{xeCJK}
% 【推荐】第一种设置
% XeLaTeX在编译时会自动选择思源宋体的bold部分
% 不设置monofont，因为有些字符在其他指定字符的出现变形，不如默认字体美观
% \setCJKmainfont{思源宋体 CN}
% \setCJKsansfont{思源黑体 CN}
% \setmainfont{思源宋体 CN}
% \setsansfont{思源黑体 CN}
% 【列示】第二种设置
% \setCJKmainfont{思源宋体}[BoldFont = 思源黑体 Regular]
%\setCJKmonofont{思源黑体 Light}
% \setCJKsansfont{思源黑体 Regular}
% \setmainfont{IBM Plex Serif}
% \setmonofont{IBM Plex Mono}
% \setsansfont{IBM Plex Sans}
```
取消相应的注释即可使用。

## image是默认的插图文件夹

## book.ps1
- 方式1：选中book.ps1文件，鼠标右击，选择“使用Powershell运行”，然后得到pdf文件

![](markdown2pdf/image/ps.jpg)

- 方式2：打开[VS Code](https://code.visualstudio.com)，点击运行；VS Code需要安装[Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner)

![](markdown2pdf/image/vscode.jpg)

## 字体下载
- 思源宋体：https://github.com/adobe-fonts/source-han-serif/blob/master/README-CN.md
- 思源黑体：https://github.com/adobe-fonts/source-han-sans/blob/master/README-CN.md

## 更多描述
[简单的笔记脚本：从markdown到pdf](https://zhuanlan.zhihu.com/p/31982147)

## 演示时运行环境
- Win 10
- pandoc
- TeX Live

### PSVersionTable
```PowerShell
PS C:\Users\cheng> $PSVersionTable

Name                           Value
----                           -----
PSVersion                      5.1.17134.48
PSEdition                      Desktop
PSCompatibleVersions           {1.0, 2.0, 3.0, 4.0...}
BuildVersion                   10.0.17134.48
CLRVersion                     4.0.30319.42000
WSManStackVersion              3.0
PSRemotingProtocolVersion      2.3
SerializationVersion           1.1.0.1
```

### xelatex --version
```PowerShell
PS C:\Users\cheng> xelatex --version
XeTeX 3.14159265-2.6-0.99999 (TeX Live 2018/W32TeX)
kpathsea version 6.3.0
Copyright 2018 SIL International, Jonathan Kew and Khaled Hosny.
There is NO warranty.  Redistribution of this software is
covered by the terms of both the XeTeX copyright and
the Lesser GNU General Public License.
For more information about these matters, see the file
named COPYING and the XeTeX source.
Primary author of XeTeX: Jonathan Kew.
Compiled with ICU version 61.1; using 61.1
Compiled with zlib version 1.2.11; using 1.2.11
Compiled with FreeType2 version 2.9.0; using 2.9.0
Compiled with Graphite2 version 1.3.11; using 1.3.11
Compiled with HarfBuzz version 1.7.6; using 1.7.6
Compiled with libpng version 1.6.34; using 1.6.34
Compiled with poppler version 0.63.0
Compiled with fontconfig version 2.13.0; using 2.13.0

### pandoc --version
```PowerShell
PS C:\Users\cheng> pandoc --version
pandoc.exe 2.1.1
Compiled with pandoc-types 1.17.3, texmath 0.10.1, skylighting 0.6
Default user data directory: C:\Users\cheng\AppData\Roaming\pandoc
Copyright (C) 2006-2018 John MacFarlane
Web:  http://pandoc.org
This is free software; see the source for copying conditions.
There is no warranty, not even for merchantability or fitness
for a particular purpose.
```

### PowerShell Core
PS1文件在PowerShell Core环境下亦可运行。
```PowerShell
PS C:\Users\cheng> $PSVersionTable

Name                           Value
----                           -----
PSVersion                      6.1.0-preview.2
PSEdition                      Core
GitCommitId                    v6.1.0-preview.2
OS                             Microsoft Windows 10.0.17134
Platform                       Win32NT
PSCompatibleVersions           {1.0, 2.0, 3.0, 4.0...}
PSRemotingProtocolVersion      2.3
SerializationVersion           1.1.0.1
WSManStackVersion              3.0
```
