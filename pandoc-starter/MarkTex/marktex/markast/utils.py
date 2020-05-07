import re,os,shutil,imghdr
from marktex import config
from urllib.parse import urljoin
from hashlib import md5
from pylatex import NoEscape

re_command = re.compile(r"%$")

re_toc = re.compile(r"^@?\[[Tt][Oo][Cc]\]") # 目录
re_maketitle = re.compile(r"^@?\[maketitle\]") # 目录
re_section = re.compile(r"^(#+)(.*)") # 章节
re_multibox = re.compile(r"^ ?\[([x√]?)\] (.*)") # 章节
# re_section = re.compile(r"^(#+)(.*)") # 章节

re_bold = re.compile(r"\*\*([^*\n]*)\*\*") # 加粗
re_italic = re.compile(r"\*([^*\n]*)\*") # 斜体
re_delete = re.compile(r"~~([^~\n]*)~~") # 删除线
re_underline = re.compile(r"__([^_\n]*)__") # 下划线

re_xml = re.compile(r"<([^/>\n]+)>([^<>]*)</([^/>\n]+)>") # xml标签（只支持一级标签

re_list = re.compile("^ *[-+] *(.*)") # 列表
re_enum = re.compile("^ *[0-9]+\. *(.*)") # enum

re_ilink = re.compile("(!?)\[([^[\n]*)\]\(([^(\n]*)\)") # link and image
re_image = re.compile("^!\[([^[\n]*)\]\(([^(\n]*)\)") # image

re_quote = re.compile("^>(.*)") # quote
re_quote_envi = re.compile("^>* *([^ ]*)")
re_quote_flag = re.compile("^>*")
re_split_rule = re.compile("^-{3,}\n?$")

re_footnote = re.compile(r"\[\^([^[^]+)\]") #footnote
re_footnote_tail = re.compile(r"^\[\^(.+)\]:(.*)") #footnote

re_table = re.compile("^\|(.*\|)+") # table
re_table_content = re.compile(r"(?=\|([^|\n]*)\|)")

re_incode = re.compile("`([^`\n]*)`") # inline code
re_code = re.compile("^ *```(.*)") # code

re_informula = re.compile("\$([^$\n]*)\$") # inline formula
re_formula = re.compile("^\$\$") # formula
re_formula_tail = re.compile("\$\$$") # formula

re_all = re.compile(r"(\*\*[^*]*\*\*|" #bold
                    r"\*[^*]*\*|" #italic
                    r"~~[^~]*~~|" #
                    r"<[^/>\n]+>[^<>]*</[^/>\n]+>|" #
                    r"__[^_]*__|" #deleteline
                    r"\[\^[^[^]+\]|" #footnote
                    r"!?\[[^[\n]*\]\([^(\n]*\)|" #link or image
                    r"`[^`\n]*`|" # code 
                    r"\$[^$\n]*\$)") # formula


_request_headers_dict = {"User-Agent" : "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9.1.6) ",
  "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  "Accept-Language" : "en-us",
  "Connection" : "keep-alive",
  "Accept-Charset" : "GB2312,utf-8;q=0.7,*;q=0.7"
                         }


class XMLTool:

    def analyse(self,s):
        from marktex.markast.xmls import xml_dict

        match_xml = re.search(re_xml,s)
        if match_xml is None:
            raise Exception(f"Match error {s}")

        tagl,content,tagr = match_xml.group(1),match_xml.group(2),match_xml.group(3)
        if tagl != tagr:
            raise Exception(f"Xml format Error {s}")

        res = xml_dict[tagl](tagl,content)
        return res


class ScanTool:
    @staticmethod
    def isBlank(line):
        return len(line.strip()) == 0

    @staticmethod
    def isToc(line):
        return re.search(re_toc, line)

    @staticmethod
    def isMakeTitle(line):
        return re.search(re_maketitle,line.lower())

    @staticmethod
    def isSplitLine(line):
        return re.search(re_split_rule,line)

    @staticmethod
    def isSection(line):
        return re.search(re_section, line)

    @staticmethod
    def isCode(line):
        return re.search(re_code, line)

    @staticmethod
    def isFormula(line):
        return re.search(re_formula, line)

    @staticmethod
    def isFormulaTail(line):
        return re.search(re_formula_tail, line)

    @staticmethod
    def isList(line):
        return re.search(re_list, line)

    @staticmethod
    def isEnum(line):
        return re.search(re_enum, line)

    @staticmethod
    def isMultiBox(line):
        return re.search(re_multibox, line)

    @staticmethod
    def isQuote(line):
        return re.search(re_quote, line)

    @staticmethod
    def isTable(line):
        return re.search(re_table, line)

    @staticmethod
    def isFootTail(line):
        return re.search(re_footnote_tail, line)

    @staticmethod
    def isImage(line):
        return re.search(re_image, line)

class MatchTool:
    @staticmethod
    def initial(line:str)->list:
        sub_line, count = re.subn(re_all,
                                  lambda s: f"@@@{s.group(0)}@@@",
                                  line)
        buffer = sub_line.split("@@@")

        return buffer

    @staticmethod
    def match_bold(token):
        return re.search(re_bold, token)

    @staticmethod
    def match_italic(token):
        return re.search(re_italic, token)

    @staticmethod
    def match_deleteline(token):
        return re.search(re_delete, token)

    @staticmethod
    def match_underline(token):
        return re.search(re_underline, token)

    @staticmethod
    def match_xml(token):
        return re.search(re_xml,token)

    @staticmethod
    def match_footnote(token):
        return re.search(re_footnote, token)

    @staticmethod
    def match_incode(token):
        return re.search(re_incode, token)

    @staticmethod
    def match_informula(token):
        return re.search(re_informula, token)

    @staticmethod
    def match_ilink(token):
        return re.search(re_ilink, token)

class ExtractTool:
    @staticmethod
    def codeType(raw):
        match_start = re.search(re_code, raw)
        return match_start.group(1)

    @staticmethod
    def tableLine(raw):
        table_line = re.findall(re_table_content, raw)
        table_line = [LineParser.fromNormal(content) for content in table_line]
        return table_line

    @staticmethod
    def multibox(raw):
        check_type = {
            "√":2,
            "x":1,
            "":0,
        }

        match = re.search(re_multibox,raw)
        return (check_type[match.group(1)],LineParser.fromNormal(match.group(2)))

    @staticmethod
    def section(raw):
        match_line = re.search(re_section, raw)
        level = len(match_line.group(1))
        content = match_line.group(2).strip()
        return level,LineParser.fromNormal(content)

    @staticmethod
    def image(raw):
        match = re.search(re_ilink, raw)
        desc = match.group(2)
        link = match.group(3)
        return desc,link

    @staticmethod
    def bold(raw):
        match = re.search(re_bold, raw)
        return match.group(1).strip()

    @staticmethod
    def italic(raw):
        match = re.search(re_italic, raw)
        return match.group(1).strip()

    @staticmethod
    def deleteline(raw):
        match = re.search(re_delete, raw)
        return match.group(1).strip()

    @staticmethod
    def underline(raw):
        match = re.search(re_underline, raw)
        return match.group(1).strip()

    @staticmethod
    def incode(raw):
        match = re.search(re_incode, raw)
        return match.group(1).strip()

    @staticmethod
    def informula(raw):
        match = re.search(re_informula, raw)
        return match.group(1).strip()

    @staticmethod
    def hyperlink(raw):
        match = re.search(re_ilink, raw)
        desc = match.group(2)
        link = match.group(3)
        link = urljoin("http:",link) #自动添加http，一般网站即使应该是https也没有影响
        return desc,link

    @staticmethod
    def footnote(raw):
        match = re.search(re_footnote, raw)
        return  match.group(1).strip()

    @staticmethod
    def footnote_tail(raw):
        match = re.search(re_footnote_tail, raw)
        key = match.group(1).strip()
        content = match.group(2).strip()
        return key,content

class SymbolTool:
    _greece = {
        "𝛼":r"\alpha",
        "𝛽":r"\beta",
        "Γ":r"\Gamma",
        "𝛾":r"\gamma",
        "Δ":r"\Delta",
        "𝛿":r"\delta",
        "𝜖":r"\epsilon",
        "𝜁":r"\zeta",
        "𝜂":r"\eta",
        "Θ":r"\Theta",
        "𝜃":r"\theta",
        "𝜄":r"\iota",
        "𝜅":r"\kappa",
        "Λ":r"\Lambda",
        "𝜆":r"\lambda",
        "𝜇":r"\mu",
        "𝜈":r"\nu",
        "Ξ":r"\Xi",
        "𝜉":r"\xi",
        "𝜊":r"\omicron",
        "Π":r"\Pi",
        "𝜋":r"\pi",
        "𝜌":r"\rho",
        "Σ":r"\Sigma",
        "𝜎":r"\sigma",
        "𝜏":r"\tau",
        "Υ":r"\Upsilon",
        "𝜐":r"\upsilon",
        "Φ":r"\Phi",
        "𝜙":r"\phi",
        "𝜒":r"\chi",
        "Ψ":r"\Psi",
        "𝜓":r"\psi",
        "Ω":r"\Omega",
        "𝜔":r"\omega",
    }
    _op = {
        "±": r"\pm",
        "×": r"\times",
        "÷": r"\div",
        "∣": r"\mid",
        "∤": r"\nmid",
        "⋅": r"\cdot",
        "∘": r"\circ",
        "∗": r"\ast",
        "⨀": r"\bigodot",
        "⨂": r"\bigotimes",
        "⨁": r"\bigoplus",
        "≤": r"\leq",
        "≥": r"\geq",
        "≠": r"\neq",
        "≈": r"\approx",
        "≡": r"\equiv",
        "∑": r"\sum",
        "∏": r"\prod",
        "∐": r"\coprod",
        "∅": r"\emptyset",
        "∈": r"\in",
        "∉": r"\notin",
        "⊂": r"\subset",
        "⊃": r"\supset",
        "⊆": r"\subseteq",
        "⊇": r"\supseteq",
        "⋂": r"\bigcap",
        "⋃": r"\bigcup",
        "⋁": r"\bigvee",
        "⋀": r"\bigwedge",
        "⨄": r"\biguplus",
        "⨆": r"\bigsqcup",
        "⊥": r"\bot",
        "∠": r"\angle",
        "′": r"\prime",
        "∫": r"\int",
        "∬": r"\iint",
        "∭": r"\iiint",
        "∮": r"\oint",
        "∞": r"\infty",
        "∇": r"\nabla",
        "∵": r"\because",
        "∴": r"\therefore",
        "∀": r"\forall",
        "∃": r"\exists",
        "≯": r"\not>",
        "⊄": r"\not\subset",
    }
    @staticmethod
    def parse(s):
        if s in SymbolTool._greece:
            return NoEscape("||"+SymbolTool._greece[s])
        elif s in SymbolTool._op:
            return NoEscape(SymbolTool._op[s])

        return NoEscape(s)

#     http://3iter.com/2015/10/14/Mathjax%E4%B8%8ELaTex%E5%85%AC%E5%BC%8F%E7%AE%80%E4%BB%8B/#mjx-eqn-1-1

class CleanTool:
    @staticmethod
    def clean_comment(doc:str):
        lines = doc.split("\n")
        res = []
        for line in lines:
            line = re.sub(re_command,"",line)
            res.append(line)
        return "\n".join(res)


    @staticmethod
    def clean_itemize(raw):
        return re.sub(re_list,lambda s:f"{s.group(1)}",raw)

    @staticmethod
    def clean_enumerate(raw):
        return re.sub(re_enum, lambda s: f"{s.group(1)}", raw)

    @staticmethod
    def get_multibox_flag(raw):
        return True

    @staticmethod
    def clean_quotes(raw):
        if isinstance(raw,list):
            return [re.sub(re_quote_envi, lambda x: f"{x.group(1)}", line) for line in raw]
        elif isinstance(raw,str):
            return re.sub(re_quote_envi,lambda s:f"{s.group(1)}",raw)

class LineParser:

    @staticmethod
    def fromNormal(line):
        from marktex.markast.line import TokenLine
        from marktex.markast.token import Bold, Italic, Footnote, InCode, InFormula, Hyperlink, InImage, Token,DeleteLine,UnderLine

        buffer = MatchTool.initial(line)
        tline = TokenLine()

        for token in buffer:

            match = MatchTool.match_bold(token)
            if match is not None:
                token = Bold(token)
                tline.append(token)
                continue

            match = MatchTool.match_xml(token)
            if match is not None:
                token = XMLTool().analyse(token)
                tline.append(token)
                continue

            match = MatchTool.match_italic(token)
            if match is not None:
                token = Italic(token)
                tline.append(token)
                continue

            match = MatchTool.match_deleteline(token)
            if match is not None:
                token = DeleteLine(token)
                tline.append(token)
                continue

            match = MatchTool.match_underline(token)

            if match is not None:
                token = UnderLine(token)
                tline.append(token)
                continue

            match = MatchTool.match_footnote(token)
            if match is not None:
                token = Footnote(token)
                tline.append(token)
                continue

            match = MatchTool.match_incode(token)
            if match is not None:
                token = InCode(token)
                tline.append(token)
                continue

            match = MatchTool.match_informula(token)
            if match is not None:
                token = InFormula(token)
                tline.append(token)
                continue

            match = MatchTool.match_ilink(token)
            if match is not None:
                if len(match.group(1)) == 0:  # 无!，是链接
                    token = Hyperlink(token)
                    tline.append(token)
                else:
                    token = InImage(token)
                    tline.append(token)
                continue

            token = Token(token)
            tline.append(token)

        return tline

class ImageTool:
    @staticmethod
    def equal(a,b):
        return os.path.getsize(a) == os.path.getsize(b) and \
               os.path.getmtime(a) == os.path.getmtime(b)


    @staticmethod
    def hashmove(pref:str, fdir:str)->str:
        pref = os.path.abspath(pref)
        size = os.path.getsize(pref)
        mtime = os.path.getmtime(pref)
        mmd = md5()
        mmd.update(str(size+mtime).encode())

        _,ext = os.path.splitext(pref)
        if ext.lower() not in ["jpg","png"]:
            ext = "png"
            import matplotlib.pyplot as plt
            pimg = plt.imread(pref)
            newf = os.path.join(fdir, f"{mmd.hexdigest()}.{ext}")
            newf = os.path.abspath(newf)
            plt.imsave(newf,pimg)
            newf = newf.replace("\\", "/")
            return newf


        newf = os.path.join(fdir,f"{mmd.hexdigest()}{ext}")
        newf = os.path.abspath(newf)

        if os.path.exists(newf) and ImageTool.equal(pref, newf):
            print(f"Have cache, checked.")
            newf = newf.replace("\\", "/")
            return newf
        else:
            shutil.copy2(pref, newf)


        print(f"Image is local file, move it \tfrom:{pref}\tto:{newf}")
        newf = newf.replace("\\","/")
        return newf


    @staticmethod
    def verify(url:str,fdir:str,rel_path):
        '''
        如果是本地图片，那么就将其hash后移动到fdir中，hash值与文件大小、修改时间有关
        如果是网络图片，那么直接根据url得到hash值，下载到fdir中
        :param url:
        :param fdir:
        :return:
        '''
        print(f"\rCheck Image:{url}.")
        os.makedirs(fdir, exist_ok=True)
        path_like = os.path.join(rel_path,url)
        print(path_like)
        if os.path.exists(path_like) and os.path.isfile(path_like):
            return ImageTool.hashmove(path_like ,fdir)


        # from urllib.request import urlretrieve
        import requests
        mmd = md5()
        mmd.update(url.encode())

        fpre = os.path.join(f"{mmd.hexdigest()}")

        fs = os.listdir(fdir)
        prefs = [os.path.splitext(f)[0] for f in fs]

        # fname = os.path.abspath(fname)
        if fpre in prefs:
            print(f"Have cache, checked.")
            i = prefs.index(fpre)
            fname = os.path.join(fdir,fs[i])
            return fname
        else:
            print("Image have't download, downloading...")

        for i in range(config.image_download_retry_time):
            fs = os.listdir(fdir)
            prefs = [os.path.splitext(f)[0] for f in fs]

            # fname = os.path.abspath(fname)
            if fpre in prefs:
                print(f"Have cache, checked.")
                i = prefs.index(fpre)
                fname = os.path.join(fdir, fs[i])
                return fname

            try:
                response = requests.get(url,headers=_request_headers_dict,timeout=5,stream=True)
                if response.status_code == 200:
                    ext = imghdr.what(None, response.content)
                    if ext not in ["jpg","png"]:
                        tempf = os.path.join(fdir, f"temp.{ext}")
                        with open(tempf, "wb") as w:
                            w.write(response.content)
                        import matplotlib.pyplot as plt
                        tmpi = plt.imread(tempf)

                        fname = os.path.join(fdir, f"{fpre}.png")
                        plt.imsave(fname,tmpi)
                        return fname
                    else:
                        fname = os.path.join(fdir, f"{fpre}.{ext}")
                        with open(fname, "wb") as w:
                            w.write(response.content)

                        return fname
            except:
                print(f"\r\ttimeout retry {i+1}/{config.image_download_retry_time}, "
                      f"you can manually download and save it in {fdir} with name {fpre}.[ext]",end="\0",flush=True)

        raise Exception(f"Error when dowanlod:{url}")

        # except:
        #     print(f"Error when download:{url}.")

