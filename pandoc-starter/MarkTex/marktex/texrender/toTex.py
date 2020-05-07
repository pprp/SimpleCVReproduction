from marktex.markast.document import Document

from marktex.markast.environment import *
from marktex.markast.line import *
from marktex.markast.token import *
from marktex.markast.xmls import *
from marktex.markast.parser import Scanner
from marktex.markast.utils import ImageTool,CleanTool
from marktex.texrender.texutils import *
from marktex import config

from pylatex.utils import bold,italic,escape_latex
from pylatex import NoEscape,Center
from pylatex import Document as TDoc,Section as TSection,Subsection,Subsubsection
from pylatex.section import Paragraph as TParagraph,Subparagraph
from pylatex import Itemize as TItem,Enumerate as TEnum,Tabular,Math

import os
# \usepackage[colorlinks=false,urlbordercolor=linkgray,pdfborderstyle={/S/U/W 1}]{hyperref}
class MarkTex(TDoc):
    def __init__(self,doc:Document,input_dir,output_dir = None,texconfig = None,subdoc = False,templete = None):
        super().__init__("", documentclass="article",
                         inputenc=None, fontenc=None, lmodern=False, textcomp=False)
        self.subdoc = subdoc

        if texconfig is None:
            texconfig = config
        self.config = texconfig

        self.input_dir = input_dir
        if output_dir is None:
            output_dir = "./"

        image_dir = os.path.join(output_dir,"images")

        self.output_dir = output_dir
        self.image_dir = os.path.abspath(image_dir)
        self.doc = doc
        self.has_toc = False

        if templete is None:
            with open(config.marktemp_path,encoding="utf-8") as f:
                self.preamble.append(NoEscape("".join(f.readlines())))

    @staticmethod
    def convert_file(fpath, output_dir=None):
        '''

        :param fpath:markdown文件的目录
        :param image_dir: markdown中的网络图片和本地图片在转换中都会被统一哈希命名并输出到一个目录
            默认是markdown文件所在的目录下的"./images"下
        :return:
        '''
        fpre,_ = os.path.split(fpath)
        if output_dir is None:
            output_dir = fpre
        os.makedirs(output_dir,exist_ok=True)

        doc = Scanner.analyse_file(fpath)

        input_dir,_ = os.path.split(fpath)
        mark = MarkTex(doc,input_dir=input_dir,output_dir=output_dir)
        mark.convert()
        return mark

    def convert(self):
        doc = self.doc
        if doc.has_maketitle:
            self.append(maketitle())

        if doc.has_toc and not self.subdoc:
            self.append(tablecontent())

        for i,envi in enumerate(doc.content):
            print(f"\rConverting...{i*100/len(doc.content):.3f}%.",end="\0",flush=True)
            if isinstance(envi,Quote):
                envi = self.fromQuote(envi)
            elif isinstance(envi,Paragraph):
                envi = self.fromParagraph(envi)
            elif isinstance(envi,Itemize):
                envi = self.fromItemize(envi)
            elif isinstance(envi,Enumerate):
                envi = self.fromEnumerate(envi)
            elif isinstance(envi,Formula):
                envi = self.fromFormula(envi)
            elif isinstance(envi,Code):
                envi = self.fromCode(envi)
            elif isinstance(envi,Table):
                envi = self.fromTable(envi)
            elif isinstance(envi,MultiBox):
                envi = self.fromMultiBox(envi)
            else:
                raise Exception(f"Doc error {envi},{envi.__class__.__name__}")
            self.append(envi)
        print(f"\rConverting...100%.")

    def fromToken(self,s:Token):
        s = escape_latex(s.string)
        return NoEscape(s)

    def fromBold(self,s:Bold):
        return bold(s.string)
    
    def fromItalic(self,s:Italic):
        return italic(s.string)

    def fromDeleteLine(self,s:DeleteLine):
        s = escape_latex(s.string)
        return NoEscape(rf"\sout{{{s}}}")

    def fromUnderLine(self,s:UnderLine):
        s = escape_latex(s.string)
        return NoEscape(rf"\underline{{{s}}}")

    def fromInCode(self,s:InCode):
        s = escape_latex(s.string)
        return NoEscape(rf"\inlang{{\small{{{s}}}}}")
    
    def fromInFormula(self,s:InFormula):
        return NoEscape(f" ${s.string}$ ")
    
    def fromHyperlink(self,s:Hyperlink):
        desc,link = escape_latex(s.desc),s.link
        return NoEscape(rf"\href{{{link}}}{{{desc}}}")

    def fromFootnote(self,s:Footnote):
        s = self.doc.footnote[s.label]
        s = escape_latex(s)
        return NoEscape(rf"\footnote{{{s}}}")
    
    def fromInImage(self,s:InImage):
        link = s.link
        link = ImageTool.verify(link, self.image_dir, self.input_dir)
        # os.chdir(cur_dir)

        if config.give_rele_path:
            link = os.path.relpath(link, self.output_dir)

        link = link.replace("\\", "/")

        return NoEscape(rf"\raisebox{{-0.5mm}}{{\includegraphics[height=1em]{{{link}}}}}")

    def fromSection(self,s:Section):
        level,content = s.level,s.content
        content = self.fromTokenLine(s.content)
        if s.level == 1:
            return TSection(content,label=False)
        elif level == 2:
            return Subsection(content,label=False)
        elif level == 3:
            return Subsubsection(content,label=False)
        elif level == 4:
            return TParagraph(content,label=False)
            # return NoEscape(r"\noindent{{\large\textbf{{{}}}}}".format(content))
            # TODO 使用paragraph还需要一些其他的包括字体在内的设置
            # return NoEscape(rf"\paragraph{{\textbf{{{content}}}}}\\")
        elif level == 5:
            # return NoEscape(r"\noindent{{\textbf{{{}}}}}".format(content.strip()))
            return Subparagraph(content,label=False)

    def fromImage(self,s:Image):
        # cur_dir = os.getcwd() #markdown的相对路径，一定是针对那个markdown的，
        # os.chdir(self.input_dir)
        link = s.link
        link = ImageTool.verify(link,self.image_dir,self.input_dir)
        # os.chdir(cur_dir)

        if config.give_rele_path:
            link = os.path.relpath(link,self.output_dir)

        link = link .replace("\\", "/")

        c = Center()
        c.append(NoEscape(
            rf"\vspace{{\baselineskip}}"
            rf"\includegraphics[width=0.8\textwidth]{{{link}}}"
            rf"\vspace{{\baselineskip}}"))
        return c

    def fromTokenLine(self,s:TokenLine):
        tokens = s.tokens
        strs = []
        for token in tokens:
            if isinstance(token,Bold):
                token = self.fromBold(token)
            elif isinstance(token,XML):
                token = self.fromXML(token)
            elif isinstance(token,Italic):
                token = self.fromItalic(token)
            elif isinstance(token,DeleteLine):
                token = self.fromDeleteLine(token)
            elif isinstance(token,Footnote):
                token = self.fromFootnote(token)
            elif isinstance(token,UnderLine):
                token = self.fromUnderLine(token)
            elif isinstance(token,InCode):
                token = self.fromInCode(token)
            elif isinstance(token,InFormula):
                token = self.fromInFormula(token)
            elif isinstance(token,Hyperlink):
                token = self.fromHyperlink(token)
            elif isinstance(token,InImage):
                token = self.fromInImage(token)
            elif isinstance(token,Token):
                token = self.fromToken(token)
            else:
                raise Exception(f"TokenLine error {token},{token.__class__.__name__}")

            strs.append(token)

        strs = NoEscape("".join(strs))
        return strs

    def fromRawLine(self,s:RawLine):
        return NoEscape(s.s)
    
    def fromNewLine(self,s:NewLine):
        return NoEscape("\n")

    def fromParagraph(self,s:Paragraph):
        t = Text()
        # Section / NewLine / TokenLine / Image
        empty = True
        for line in s.buffer:
            if isinstance(line,Section):
                line = self.fromSection(line)
            elif isinstance(line,NewLine):
                line = self.fromNewLine(line)
            elif isinstance(line,TokenLine):
                line = self.fromTokenLine(line)
            elif isinstance(line,Image):
                line = self.fromImage(line)
            else:
                raise Exception(f"Paragraph line error {line} is {line.__class__}")
            t.append(line)
            empty = False

        if empty:
            return NoEscape("\n")
        return t

    def fromQuote(self,s:Quote):
        content = s.doc.content
        q = QuoteEnvironment()
        for envi in content:
            if isinstance(envi,Paragraph):
                envi = self.fromParagraph(envi)
            elif isinstance(envi,Table):
                envi = self.fromTable(envi)
            elif isinstance(envi,Itemize):
                envi = self.fromItemize(envi)
            elif isinstance(envi,Enumerate):
                envi = self.fromEnumerate(envi)
            elif isinstance(envi,Formula):
                envi = self.fromFormula(envi)
            elif isinstance(envi,Code):
                envi = self.fromCode(envi)
            else:
                raise Exception(f"Quote doc error:{envi},{envi.__class__.__name__}")
            q.append(envi)
            q.append(NoEscape("\n"))

        return q

    def fromItemize(self,s:Itemize):
        tokens = [self.fromTokenLine(c) for c in s.buffer]
        ui = TItem()
        for line in tokens:
            ui.add_item(line)
        return ui
    
    def fromMultiBox(self,s:MultiBox):
        cl = CheckList()
        for [ct,s] in s.lines:
            cl.add_item(ct,s)
        return cl
    
    def fromEnumerate(self,s:Enumerate):
        tokens = [self.fromTokenLine(c) for c in s.buffer]
        ui = TEnum()
        for line in tokens:
            ui.add_item(line)
        return ui

    def fromFormula(self,s:Formula):
        code = [self.fromRawLine(c) for c in s.formula]

        data = []
        for line in code:
            data.append(NoEscape(f"{line}\\\\"))

        m = Math(data=data)
        # eq = Equation()
        # for line in code:
        #     eq.append(line)
        return m

    def fromCode(self,s:Code):
        code = [self.fromRawLine(c) for c in s.code]
        c = CodeEnvironment(s.code_style,self.config)
        for line in code:
            line = line.replace("\t","    ")
            c.append(NoEscape(line))

        return c

    def fromTable(self,s:Table):
        c = Center()
        # c.append(NoEscape(r"\newlength\q"))
        c.append(
            NoEscape(
                rf"\setlength\tablewidth{{\dimexpr (\textwidth -{2*s.col_num}\tabcolsep)}}"))
        c.append(NoEscape(r"\arrayrulecolor{tablelinegray!75}"))
        c.append(NoEscape(r"\rowcolors{2}{tablerowgray}{white}"))


        ratios = s.cacu_col_ratio()
        # format = "|".join([rf"p{{{r}\textwidth}}<{{\centering}}" for r in ratios])
        format = "|".join([rf"p{{{r}\tablewidth}}<{{\centering}}" for r in ratios])
        format = f"|{format}|"

        t = Tabular(format)
        t.add_hline()
        for i,row in enumerate(s.tables):
            if i == 0:
                t.append(NoEscape(r"\rowcolor{tabletopgray}"))

            row = [self.fromTokenLine(c) for c in row]
            if i == 0:
                row = [bold(c) for c in row]

            t.add_row(row)
            t.add_hline()

        c.append(t)
        return c

    def fromXML(self,token:XML):
        if isinstance(token,XMLTitle):
            self.preamble.append(NoEscape(rf"\title{{{token.content}}}"))
            return NoEscape("")
        elif isinstance(token,XMLAuthor):
            self.preamble.append(NoEscape(rf"\author{{{token.content}}}"))
            return NoEscape("")
        elif isinstance(token,XMLSub):
            return NoEscape(rf"\textsubscript{{{token.content}}}")
        elif isinstance(token,XMLSuper):
            return NoEscape(rf"\textsuperscript{{{token.content}}}")
        elif isinstance(token,XMLInclude):
            cur_dir = os.getcwd()
            os.chdir(self.input_dir)
            if token.content.endswith("md"):
                doc = MarkTex.convert_file(token.content)
                os.chdir(cur_dir)
                return NoEscape(doc.dumps_content())
            elif token.content.endswith("tex"):
                with open(token.content,encoding="utf-8") as f:
                    lines = "".join(f.readlines())
                    os.chdir(cur_dir)
                    return NoEscape(lines)
            raise Exception("format not support")

    def dumps(self):
        string = super().dumps()
        string = CleanTool.clean_comment(string)
        return string

    def generate_tex(self, filename=None):
        '''
        输入文件名即可，保存路径在输入时已经确定好了
        :param filename:
        :return:
        '''
        filepath = os.path.join(self.output_dir,filename)
        super().generate_tex(filepath)
        print(f"File is output in {os.path.abspath(filepath)}.tex and images is in {os.path.abspath(self.image_dir)}.")
