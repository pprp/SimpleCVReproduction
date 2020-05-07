# from marktex.markast.environment import *
from .environment import *

class Document:
    paragraph = "Paragraph"
    quote = "Quote"
    itemize = "Itemize"
    enumerate = "Enumerate"
    formula = "Formula"
    code = "Code"
    table = "Table"
    multi = "Multi"

    mode_dict = {
        "Paragraph": Paragraph,
        "Quote": Quote,
        "Itemize": Itemize,
        "Enumerate": Enumerate,
        "Formula": Formula,
        "Code": Code,
        "Table": Table,
        "Multi": MultiBox,
    }

    def __init__(self):
        self.has_toc = False
        self.has_maketitle = False
        self.content = [] #list[Environment]
        self.cur = None
        self.cur_mode = None
        self.footnote = {}


    def append(self,data:Environment):
        if isinstance(data,Environment):
            self.content.append(data)

    def open_toc(self):
        self.has_toc = True

    def make_title(self):
        self.has_maketitle = True

    def new_line(self):
        cur = self.change(Document.paragraph)
        cur.new_line()

    def append_footnote_tail(self,label,desc):
        if label in self.footnote:
            raise Exception(f"dumplicate footnote:{label}")
        self.footnote[label] = desc


    def change(self,mode)->Environment:
        if mode not in Document.mode_dict:
            raise Exception(f"{mode} not in mode_dict")

        if self.cur_mode != mode:
            self._change(mode)

        return self.cur

    def finished(self):
        self.cur.close()
        self.content.append(self.cur)


    def _change(self,mode):
        if self.cur is not None:
            self.cur.close()
            self.content.append(self.cur)
        self.cur = Document.mode_dict[mode]()
        self.cur_mode = mode

    def __str__(self):
        head = "\n".join([f"{i}" for i in self.content])
        if len(self.footnote) == 0:
            return f"{head}\n"
        tail = "\n".join([f"{k}:{v}" for k,v in self.footnote.items()])
        return f"{head}\n————————\n{tail}"