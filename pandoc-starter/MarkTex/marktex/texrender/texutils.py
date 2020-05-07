from pylatex import NoEscape,Package,Command
from pylatex.base_classes import Environment,Container
from marktex import config
from marktex.texrender import texparam

class Text(Container):
    """A class that represents a section."""

    #: A section should normally start in its own paragraph
    end_paragraph = True

    #: Default prefix to use with Marker
    marker_prefix = "sec"

    #: Number the sections when the section element is compatible,
    #: by changing the `~.Section` class default all
    #: subclasses will also have the new default.
    numbering = True

    def __init__(self,**kwargs):
        """
        Args
        ----
        title: str
            The section title.
        numbering: bool
            Add a number before the section title.
        label: Label or bool or str
            Can set a label manually or use a boolean to set
            preference between automatic or no label
        """
        super().__init__(**kwargs)

    def dumps(self):
        """Represent the section as a string in LaTeX syntax.

        Returns
        -------
        str

        """
        string = '%\n' + self.dumps_content()
        return string


class Equation(Environment):
    _latex_name = "equation"

class CodeEnvironment(Environment):
    _latex_name = "langbox"

    cpp = "C++"

    code_style_dict = {
        "cpp": "C++"

    }
    def __init__(self,mode=None,texconfig = None,**kwargs):
        if texconfig is None:
            texconfig = config
        if mode is None or len(mode.strip()) == 0:
            mode = "Tex"

        if mode.lower() in CodeEnvironment.code_style_dict:
            mode = CodeEnvironment.code_style_dict[mode]

        options = [NoEscape(f"{mode.capitalize()}")]
        super().__init__(options=options, arguments=None, start_arguments=None, **kwargs)


class QuoteEnvironment(Environment):
    _latex_name = "markquote"

class TColorBox(Environment):
    _latex_name = "tcolorbox"

class CheckList(Environment):
    """A base class that represents a list."""

    #: List environments cause compile errors when they do not contain items.
    #: This is why they are omitted fully if they are empty.
    omit_if_empty = True
    _latex_name = "itemize"

    def __init__(self, *, options=None, arguments=None, start_arguments=None, **kwargs):
        super().__init__(options=options, arguments=arguments, start_arguments=start_arguments, **kwargs)
        self.packages.add(Package("pifont"))

    uncheck_item = NoEscape("$\square$")
    check_item = NoEscape(r"\rlap{\raisebox{0.3ex}{\hspace{0.4ex}\tiny \ding{52}}}$\square$")
    nocheck_iteunm = NoEscape(r"\rlap{\raisebox{0.3ex}{\hspace{0.4ex}\scriptsize \ding{56}}}$\square$")

    def add_item(self,ct:int,s):
        """Add an item to the list.

        Args
        ----
        s: str or `~.LatexObject`
            The item itself.
        """
        if ct == 0:
            self.append(Command('item',options=CheckList.uncheck_item))
        elif ct == 1:
            self.append(Command('item',options=CheckList.check_item))
        elif ct == 2:
            self.append(Command('item',options=CheckList.nocheck_iteunm))

        self.append(s)


def tablecontent():
    return NoEscape("\\tableofcontents\n\\newpage")

def maketitle():
    return NoEscape(r"\maketitle")

def print_markenv():
    with open(config.marktemp_path,encoding="utf-8") as f:
        for l in f:
            print(l)

def create_markenvf(fn):
    import shutil
    shutil.copy(config.marktemp_path,fn)