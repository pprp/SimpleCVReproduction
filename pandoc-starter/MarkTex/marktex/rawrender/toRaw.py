import os
from marktex.markast.utils import ImageTool,CleanTool
from marktex.markast.parser import Scanner
from marktex import config
from marktex.markast.document import Document

from marktex.markast.environment import *
from marktex.markast.line import *
from marktex.markast.token import *
from marktex.markast.xmls import *

class MarkRaw():

    def __init__(self, doc: Document, input_dir, output_dir=None, texconfig=None, subdoc=False):
        self.subdoc = subdoc

        if texconfig is None:
            texconfig = config
        self.config = texconfig

        self.input_dir = input_dir
        if output_dir is None:
            output_dir = "./"

        image_dir = os.path.join(output_dir, "images")

        self.output_dir = output_dir
        self.image_dir = os.path.abspath(image_dir)
        self.doc = doc
        self.has_toc = False
        self.contents = []
    def append(self,item):
        self.contents.append(item)

    @staticmethod
    def convert_file(fpath, output_dir=None):
        '''

        :param fpath:markdown文件的目录
        :param image_dir: markdown中的网络图片和本地图片在转换中都会被统一哈希命名并输出到一个目录
            默认是markdown文件所在的目录下的"./images"下
        :return:
        '''
        fpre, _ = os.path.split(fpath)
        if output_dir is None:
            output_dir = fpre
        os.makedirs(output_dir, exist_ok=True)

        doc = Scanner.analyse_file(fpath)

        input_dir, _ = os.path.split(fpath)
        mark = MarkRaw(doc, input_dir=input_dir, output_dir=output_dir)
        mark.convert()
        return mark

    def convert(self):
        doc = self.doc
        if doc.has_toc and not self.subdoc:
            pass

        for i, envi in enumerate(doc.content):
            print(f"\rConverting...{i * 100 / len(doc.content):.3f}%.", end="\0", flush=True)
            if isinstance(envi, Quote):
                envi = self.fromQuote(envi)
            elif isinstance(envi, Paragraph):
                envi = self.fromParagraph(envi)
            elif isinstance(envi, Itemize):
                envi = self.fromItemize(envi)
            elif isinstance(envi, Enumerate):
                envi = self.fromEnumerate(envi)
            elif isinstance(envi, Formula):
                envi = self.fromFormula(envi)
            elif isinstance(envi, Code):
                envi = self.fromCode(envi)
            elif isinstance(envi, Table):
                envi = self.fromTable(envi)
            elif isinstance(envi, MultiBox):
                envi = self.fromMultiBox(envi)
            else:
                raise Exception(f"Doc error {envi},{envi.__class__.__name__}")
            self.append(envi)
        print(f"\rConverting...100%.")

    def fromToken(self, s: Token):
        return s.string

    def fromBold(self, s: Bold):
        return s.string

    def fromItalic(self, s: Italic):
        return s.string

    def fromDeleteLine(self, s: DeleteLine):
        return s.string

    def fromUnderLine(self, s: UnderLine):
        return s.string

    def fromInCode(self, s: InCode):
        return s.string

    def fromInFormula(self, s: InFormula):
        return s.string

    def fromHyperlink(self, s: Hyperlink):
        desc, link = s.desc, s.link
        return f" {link},{desc} "

    def fromFootnote(self, s: Footnote):
        return s.label

    def fromInImage(self, s: InImage):
        return ""

    def fromSection(self, s: Section):
        level, content = s.level, s.content
        content = self.fromTokenLine(s.content)
        return f"{level}-{content}"

    def fromImage(self, s: Image):
        # cur_dir = os.getcwd() #markdown的相对路径，一定是针对那个markdown的，
        # os.chdir(self.input_dir)
        link = s.link
        link = ImageTool.verify(link, self.image_dir, self.input_dir)
        # os.chdir(cur_dir)

        if config.give_rele_path:
            link = os.path.relpath(link, self.output_dir)

        link = link.replace("\\", "/")
        return f" img,{link} "

    def fromXML(self, token: XML):
        return token.content

    def fromTokenLine(self, s: TokenLine):
        tokens = s.tokens
        strs = []
        for token in tokens:
            if isinstance(token, Bold):
                token = self.fromBold(token)
            elif isinstance(token, XML):
                token = self.fromXML(token)
            elif isinstance(token, Italic):
                token = self.fromItalic(token)
            elif isinstance(token, DeleteLine):
                token = self.fromDeleteLine(token)
            elif isinstance(token, Footnote):
                token = self.fromFootnote(token)
            elif isinstance(token, UnderLine):
                token = self.fromUnderLine(token)
            elif isinstance(token, InCode):
                token = self.fromInCode(token)
            elif isinstance(token, InFormula):
                token = self.fromInFormula(token)
            elif isinstance(token, Hyperlink):
                token = self.fromHyperlink(token)
            elif isinstance(token, InImage):
                token = self.fromInImage(token)
            elif isinstance(token, Token):
                token = self.fromToken(token)
            else:
                raise Exception(f"TokenLine error {token},{token.__class__.__name__}")

            strs.append(token)

        return "".join(strs)

    def fromRawLine(self, s: RawLine):
        return s.s

    def fromNewLine(self, s: NewLine):
        return "\n"

    def fromParagraph(self, s: Paragraph):
        t = []
        # Section / NewLine / TokenLine / Image
        for line in s.buffer:
            if isinstance(line, Section):
                line = self.fromSection(line)
            elif isinstance(line, NewLine):
                line = self.fromNewLine(line)
            elif isinstance(line, TokenLine):
                line = self.fromTokenLine(line)
            elif isinstance(line, Image):
                line = self.fromImage(line)
            else:
                raise Exception(f"Paragraph line error {line} is {line.__class__}")
            t.append(line)

        return "\n".join(t)

    def fromQuote(self, s: Quote):
        content = s.doc.content
        q = []
        for envi in content:
            if isinstance(envi, Paragraph):
                envi = self.fromParagraph(envi)
            elif isinstance(envi, Table):
                envi = self.fromTable(envi)
            elif isinstance(envi, Itemize):
                envi = self.fromItemize(envi)
            elif isinstance(envi, Enumerate):
                envi = self.fromEnumerate(envi)
            elif isinstance(envi, Formula):
                envi = self.fromFormula(envi)
            elif isinstance(envi, Code):
                envi = self.fromCode(envi)
            else:
                raise Exception(f"Quote doc error:{envi},{envi.__class__.__name__}")
            q.append(envi)

        return "\n".join(q)

    def fromItemize(self, s: Itemize):
        tokens = [self.fromTokenLine(c) for c in s.buffer]
        ui = []
        for line in tokens:
            ui.append(f" - {line}")
        return "\n".join(ui)

    def fromMultiBox(self, s: MultiBox):
        cl = []
        for [ct, s] in s.lines:
            cl.append(f"{ct} {s}")
        return "\n".join(cl)

    def fromEnumerate(self, s: Enumerate):
        tokens = [self.fromTokenLine(c) for c in s.buffer]
        ui = []
        for i,line in enumerate(tokens):
            ui.append(f"{i},{line}")
        return "\n".join(ui)

    def fromFormula(self, s: Formula):
        code = [self.fromRawLine(c) for c in s.formula]

        data = []
        for line in code:
            data.append(line)

        return "\n".join(data)

    def fromCode(self, s: Code):
        code = [self.fromRawLine(c) for c in s.code]
        c = []
        for line in code:
            c.append(line)
        return "\n".join(c)

    def fromTable(self, s: Table):

        t = []
        for i, row in enumerate(s.tables):
            row = [self.fromTokenLine(c) for c in row]


            t.append(" & ".join(row))

        return "\n".join(t)

    def generate_txt(self, filename=None):
        '''
        输入文件名即可，保存路径在输入时已经确定好了
        :param filename:
        :return:
        '''
        filepath = os.path.join(self.output_dir, f"{filename}.txt")
        with open(f"{filepath}","w",encoding="utf-8") as w:
            w.writelines(self.contents)
        print(f"File is output in {os.path.abspath(filepath)} and images is in {os.path.abspath(self.image_dir)}.")


