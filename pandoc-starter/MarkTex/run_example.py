from marktex.example import run_example

# run_example("./output/")


from marktex.texrender.toTex import MarkTex

from marktex.markast.parser import Scanner

doc = MarkTex.convert_file("01-Faster RCNN代码解析第一弹.md",output_dir="./output")
doc.generate_tex("example")


# doc = Scanner.analyse_file("./marktex/example/example.md")
# print(doc)

# from marktex.example import run_example
# run_example("./output/")