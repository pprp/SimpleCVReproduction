from marktex.markast.utils import ExtractTool,SymbolTool
from pylatex import NoEscape

class Token: #可以作为 行内的部分出现

    def __init__(self,s) -> None:
        super().__init__()
        self._parse_sym = True
        self.string = s
        self.initial()
        # if self._parse_sym:
        #     self._extract()

    def _extract(self):
        strlist = list(self.string)
        self.string = NoEscape("".join([SymbolTool.parse(i) for i in strlist]))

    def initial(self):
        pass

    def __str__(self) -> str:
        return self.string.__str__()

    def __len__(self):
        return len(self.string)

class Bold(Token):

    def initial(self):
        self.string = ExtractTool.bold(self.string)

    def __len__(self):
        return len(self.string)

    def __str__(self) -> str:
        return f"Bold({self.string})"

class Italic(Token):
    def initial(self):
        self.string = ExtractTool.italic(self.string)

    def __len__(self):
        return len(self.string)

    def __str__(self) -> str:
        return f"Italic({self.string})"

class InFormula(Token):
    def initial(self):
        self.string = ExtractTool.informula(self.string)

    def __str__(self) -> str:
        return f"InFormula({self.string})"

    def __len__(self):
        return len(self.string)

class InCode(Token):
    def initial(self):

        self.string = ExtractTool.incode(self.string)

    def __str__(self) -> str:
        return f"InCode({self.string})"

class Hyperlink(Token):

    def initial(self):
        self._parse_sym = False
        self.desc,self.link = ExtractTool.hyperlink(self.string)

    def __str__(self) -> str:
        return f"Hyperlink({self.desc};{self.link})"

    def __len__(self):
        return len(self.desc)

class Footnote(Token):

    def initial(self):
        self._parse_sym = True
        self.label = ExtractTool.footnote(self.string)

    def __str__(self) -> str:
        return f"Footnote({self.label})"

    def __len__(self):
        return 1

class InImage(Token):
    def initial(self):
        self._parse_sym = True
        self.desc, self.link = ExtractTool.image(self.string)
    def __str__(self) -> str:
        return f"Hyperlink({self.desc};{self.link})"

    def __len__(self):
        return len(self.desc)

class UnderLine(Token):
    def initial(self):
        self.string = ExtractTool.underline(self.string)

class DeleteLine(Token):
    def initial(self):
        self.string = ExtractTool.deleteline(self.string)