from marktex.texrender import MarkTex

doc = MarkTex.convert_file(
    r".\test_in_md\01-Faster RCNN代码解析第一弹.md", r".\test_out_pdf")
doc.generate_tex()
