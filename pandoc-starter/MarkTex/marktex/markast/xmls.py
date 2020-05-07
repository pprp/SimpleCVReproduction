from marktex.markast.token import Token

xml_dict = {}

def xml(fn):
    xml_dict[str(fn._xml_tag)] = fn
    return fn


class XML:
    _xml_tag = ""
    def __init__(self,tag:str,content:str) -> None:
        # super().__init__(content)
        self.tag = tag
        self.content = content

    def initial(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__.capitalize()}({self.content})"

    def __len__(self):
        return len(self.content)


@xml
class XMLSub(XML):
    _xml_tag = "sub"
    pass

@xml
class XMLSuper(XML):
    _xml_tag = "super"
    pass


@xml
class XMLInclude(XML):
    _xml_tag = "include"
    pass

@xml
class XMLTitle(XML):
    _xml_tag = "title"
    pass

@xml
class XMLAuthor(XML):
    _xml_tag = "author"
    pass