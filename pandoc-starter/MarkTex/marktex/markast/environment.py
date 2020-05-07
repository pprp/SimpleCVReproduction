from .utils import ExtractTool
from marktex.markast.line import RawLine,NewLine

class Environment:

    def __init__(self) -> None:
        self.buffer = []

    def append_raw(self,line):
        self.buffer.append(line)

    def last_line(self):
        return self.buffer[-1]

    def new_line(self):
        self.buffer.append(NewLine())

    def initial(self):
        '''递归的分析其接收到的内容'''
        # raise NotImplementedError("")
        pass

    def close(self):
        self.initial()

    def __str__(self):
        st = [f"{c}" for c in self.buffer]
        return "\n".join(st)

class Paragraph(Environment):
    # buffer:list[subclass of Line,include Section/NewLine/TokenLine/Image]
    def initial(self):
        super().initial()

    def __str__(self):
        st = "\n".join([f"{c}" for c in self.buffer])
        return f"Paragraph:{{\n" \
               f"{st}\n" \
               f"}}ParagraphEnd"

class Quote(Paragraph):
    # 只含有一个doc
    def initial(self):
        self.doc = self.buffer[0]

    def __str__(self):
        return f"Quote:{{\n" \
               f"{self.doc}\n" \
               f"}}QuoteEnd"

class Itemize(Environment): # 包括Itemize和Enumerate
    # buffer:list[TokenLine]
    def __str__(self):
        st = "\n".join([f" - {c}" for c in self.buffer])
        return f"Itemize:{{\n" \
               f"{st}\n" \
               f"}}ItemizeEnd"

class Enumerate(Environment): # 包括Itemize和Enumerate
    # buffer:list[TokenLine]
    def __str__(self):
        st = "\n".join([f"{i+1}.{c}" for i,c in enumerate(self.buffer)])
        return f"Enumerate:{{\n" \
               f"{st}\n" \
               f"}}EnumrateEnd"

class MultiBox(Environment):#复选框
    # buffer:list[TokenLine]

    def initial(self):
        self.lines = []

        for line in self.buffer:
            self.lines.append(ExtractTool.multibox(line))



class Formula(Environment):
    # formula:list[RawLine]
    def initial(self):
        self.buffer[0] = self.buffer[0][2:]
        self.buffer[-1] = self.buffer[0][:-2]

        self.formula = [RawLine(i) for i in self.buffer]

    def __str__(self):
        st = "\n".join(self.buffer)
        return f"Formula:{{\n" \
               f"{st}\n" \
               f"}}FormulaEnd"

class Code(Environment):
    #code:list[RawLine]


    def initial(self):
        code_style = ExtractTool.codeType(self.buffer[0])


        if len(code_style) == 0:
            code_style = "Tex"
        self.code_style = code_style

        self.code = self.buffer[1:]
        self.code = [RawLine(i) for i in self.code]
        pass

    def __str__(self):
        code = "\n".join([f"\t{c}" for c in self.code])
        return f"Code style={self.code_style}.\n" \
               f"Code content:\n" \
               f"{code}\n" \
               f"CodeEnd"

class Table(Environment):
    # 每一个元素都是TokenLine
    def initial(self):
        self.tables = []
        col_num = None
        col_max_lens = None
        for i,line in enumerate(self.buffer):
            if i == 1: #略过第二行
                continue

            row = ExtractTool.tableLine(line)
            if col_num is None:
                col_num = len(row)
            elif col_num != len(row):
                raise Exception(f"Tabel's column num must be equal,but \n{self.buffer}")

            if col_max_lens is None:
                col_max_lens = [len(i) for i in row]
            else:
                col_max_lens = [max(i,len(j)) for i,j in zip(col_max_lens,row)]

            self.tables.append(row)

        self.shape = (col_num,len(self.tables))
        self.col_max_lens = col_max_lens

    @property
    def col_num(self):
        return self.shape[0]

    def cacu_col_ratio(self):
        col_max_lens = [min(i,10) for i in self.col_max_lens]
        ratio = [i/sum(col_max_lens) for i in col_max_lens]

        return [f"{i:.3f}" for i in ratio]



    def __str__(self):
        tables = self.tables
        max_len = 0
        for line in tables:
            for col in line:
                max_len = max(max_len,len(col))
        max_len+=2

        st = []
        for line in tables:
            line = "|".join([c.__str__().ljust(max_len) for c in line])
            st.append(line)

        st.insert(1,"".ljust(self.col_num*max_len,"-"))

        st = "\n".join(st)

        return f"Table:{{shape={self.shape}\n" \
               f"{st}\n" \
               f"}}TableEnd"


