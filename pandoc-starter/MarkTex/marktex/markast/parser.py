from marktex.markast.utils import ScanTool, CleanTool, LineParser
from marktex.markast.document import Document
from marktex.markast.line import *

class Scanner:

    def __init__(self) -> None:
        super().__init__()
        self.doc = Document()
        self.lp = LineParser()

    @staticmethod
    def analyse_file(fname:str):
        with open(fname,"r",encoding="utf-8") as f:
            lines = f.readlines()

        return Scanner().analyse(lines)

    def analyse(self,lines:list):
        lines = [i.strip("\n") for i in lines]

        index = 0
        size = len(lines)
        doc = self.doc
        lp = self.lp
        while index<size:
            print(f"\rExtract Markdown Struct:{index*100/size:.3f}%",end="\0",flush=True)
            line = lines[index].strip()
            if ScanTool.isBlank(line):#由doc处理√
                doc.new_line()
            elif ScanTool.isSplitLine(line):
                pass
            elif ScanTool.isToc(line):#由doc处理√
                doc.open_toc()
            elif ScanTool.isMakeTitle(line):#由doc处理√
                doc.make_title()
            elif ScanTool.isSection(line):#在Line内处理√
                cur = doc.change(Document.paragraph)
                cur.append_raw(Section(line))
            elif ScanTool.isCode(line):#在环境内处理√
                cur = doc.change(Document.code)
                cur.append_raw(lines[index])
                index += 1
                while index<size and not ScanTool.isCode(lines[index]):
                    cur.append_raw(lines[index])
                    index+=1
            elif ScanTool.isFormula(line):#在环境内处理√
                cur = doc.change(Document.formula)
                cur.append_raw(lines[index])
                tmp = index
                if len(lines[index]) == 2:
                    index += 1
                while index<size and not ScanTool.isFormulaTail(lines[index]):
                    cur.append_raw(lines[index])
                    index+=1
                if index < size and tmp != index:
                    cur.append_raw(lines[index])
            elif ScanTool.isList(line):#直接处理√
                cur = doc.change(Document.itemize)
                line = CleanTool.clean_itemize(line)
                tokens = lp.fromNormal(line)
                cur.append_raw(tokens)
            elif ScanTool.isEnum(line):#直接处理√
                cur = doc.change(Document.enumerate)
                line = CleanTool.clean_enumerate(line)
                tokens = lp.fromNormal(line)
                cur.append_raw(tokens)
            elif ScanTool.isMultiBox(line):#TODO 在环境内处理
                cur = doc.change(Document.multi)
                # newline = CleanTool.clean_multibox(line)
                # tokens = lp.fromNormal(newline)
                cur.append_raw(line)
            elif ScanTool.isQuote(line):#在方法内，转换为doc处理
                cur = doc.change(Document.quote)
                temp = index
                while temp<size and not ScanTool.isBlank(lines[temp]):
                    temp += 1
                qdoc = self.analyse_quote(lines[index:temp+1])
                cur.append_raw(qdoc)
                index=temp
                doc.new_line()
            elif ScanTool.isTable(line):#在环境内处理√
                cur = doc.change(Document.table)
                cur.append_raw(line)
            elif ScanTool.isFootTail(line):#在doc方法里处理
                key, content = ExtractTool.footnote_tail(line)
                doc.append_footnote_tail(key,content)
            elif ScanTool.isImage(line):#在Line内处理
                cur = doc.change(Document.paragraph)
                cur.append_raw(Image(line))
            else:#直接处理
                cur = doc.change(Document.paragraph)
                cur.append_raw(lp.fromNormal(line))
            index+=1
        print(f"\rExtract Markdown Struct:100%.Finished",flush=True)
        doc.finished()
        print("Finished")
        return doc

    def analyse_quote(self,lines: list):
        lines = CleanTool.clean_quotes(lines)

        index = 0
        size = len(lines)
        doc = Document()
        lp = self.lp
        while index < size:
            line = lines[index].strip()

            if ScanTool.isBlank(line):#由doc处理
                doc.new_line()
            elif ScanTool.isCode(line):# 在环境内自行处理
                cur = doc.change(Document.code)
                cur.append_raw(lines[index])
                index += 1
                while index < size and not ScanTool.isCode(lines[index]):
                    cur.append_raw(lines[index])
                    index += 1
            elif ScanTool.isFormula(line):# 在环境内自行处理
                cur = doc.change(Document.formula)
                cur.append_raw(lines[index])
                tmp = index
                if len(lines[index]) == 2:
                    index += 1
                while index < size and not ScanTool.isFormulaTail(lines[index]):
                    cur.append_raw(lines[index])
                    index += 1
                if index < size and tmp != index:
                    cur.append_raw(lines[index])
            elif ScanTool.isList(line): #直接处理
                cur = doc.change(Document.itemize)
                line = CleanTool.clean_itemize(line)
                tokens = lp.fromNormal(line)
                cur.append_raw(tokens)
            elif ScanTool.isEnum(line): #直接处理
                cur = doc.change(Document.enumerate)
                line = CleanTool.clean_enumerate(line)
                tokens = lp.fromNormal(line)
                cur.append_raw(tokens)
            elif ScanTool.isTable(line):#在环境内自行处理
                cur = doc.change(Document.table)
                cur.append_raw(line)
            elif ScanTool.isFootTail(line):
                key, content = ExtractTool.footnote_tail(line)
                doc.append_footnote_tail(key, content)
            elif ScanTool.isImage(line): #在Line内自行处理
                cur = doc.change(Document.paragraph)
                cur.append_raw(Image(line))
            else:#直接处理
                cur = doc.change(Document.paragraph)
                cur.append_raw(lp.fromNormal(line))

            index += 1

        doc.finished()
        return doc

