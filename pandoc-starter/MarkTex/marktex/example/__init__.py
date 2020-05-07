from marktex.texrender.toTex import MarkTex
import os

def run_example(output_dir,fname = None):
    path,_ = os.path.split(__file__)
    md_file = os.path.join(path,"example.md")

    os.makedirs(output_dir,exist_ok=True)
    doc = MarkTex.convert_file(md_file, output_dir=output_dir)

    if fname is None:
        fname = "example"
    
    doc.generate_tex(fname)
    

