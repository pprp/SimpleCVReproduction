from .utils import ExtractTool
from .token import Footnote

class Line: #表示单独成行,默认不进行任何处理

    def __init__(self,s) -> None:
        self.s = s
        self.initial(s)

    def initial(self,s):
        pass

    def __str__(self) -> str:
        return f"{self.s}"


class Section(Line):#标题单独成行
    #level:int
    #content:TokenLine
    def initial(self, s):
        self.level,self.content = ExtractTool.section(s)

    def __str__(self) -> str:
        return f"Section: Level={self.level},content={self.content}"

class Image(Line):#图片单独成行
    #desc:str
    #link:str
    def initial(self, s):
        self.desc,self.link = ExtractTool.image(s)

    def __str__(self):
        return f"Image:[{self.desc}]({self.link})"

class TokenLine:#普通的一行
    def __init__(self):
        self.tokens = []
        self.footnote = []

    def append(self,data):
        if isinstance(data,Footnote):
            self.footnote.append(data)
        self.tokens.append(data)

    def __str__(self):
        return "".join([i.__str__() for i in self.tokens])

    def __len__(self):
        return sum([len(i) for i in self.tokens])

class RawLine(Line):#只在代码和公式环境中出现
    pass

class NewLine(Line):#用于换行


    def __init__(self, s=None) -> None:
        super().__init__("")

    def __str__(self) -> str:
        return "NewLine()"