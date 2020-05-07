import argparse,sys,os
from argparse import RawDescriptionHelpFormatter

APP_DESC="""
MarkTex is used to convert markdown document into tex format.

You can choose the output dir from:
- output to the dir of every md file, the default option.
- output to a single dir use -o "path" option.
- assign every path use -e "path1" "path2" option.    
"""

example = '''
e.g:
Output to the dir the md file is:\n\tmarktex a.md ../b.md /dir/c.md ...
Assign every path:\n\tmarktex a.md b.md ... -o "path"
Assign every path (be sure the number of the dir options must be equal to markdown files):\n\tmarktex a.md b.md ... -e "patha" "pathb" ...
'''
if len(sys.argv) == 1:
    sys.argv.append('--help')
parser = argparse.ArgumentParser(
    prog="marktex",
    description="test desc",
    # usage=APP_DESC,
    epilog=example,formatter_class=RawDescriptionHelpFormatter)

parser.add_argument('mdfiles', metavar='mdfiles', type=str, nargs='+',
                    help='place markdown path')
parser.add_argument('-o',
                    '--output',
                    dest="out",
                    action="store",
                    type=str,
                    default=None,
                    help="指定统一路径")

parser.add_argument('-r',
                    '--raw',
                    dest="raw_text",
                    action="store_true",
                    default=False,
                    help="输出纯字符串")

parser.add_argument('-e','--each',help="为每个文件分配路径",nargs="*")
args = parser.parse_args()

print(args)

every = args.each
mdfiles = args.mdfiles
output = args.out
output_paths = []
raw = args.raw_text
if every is not None:
    if len(every) != len(mdfiles):
        print("you ues -e option, the number of outputdirs must be equal to markdown files.")
        exit(1)
    output_paths = every
elif output is not None:
    output_paths = [output]*len(mdfiles)
else:
    for mdfile in mdfiles:
        mdfile = os.path.abspath(mdfile)
        mdpath,fname = os.path.splitext(mdfile)
        output_paths.append(mdpath)

from marktex.texrender.toTex import MarkTex
from marktex.rawrender.toRaw import MarkRaw

for mdfile,opath in zip(mdfiles,output_paths):
    _,fname = os.path.split(mdfile)
    fpre,_ = os.path.splitext(fname)
    if raw:
        fpre, _ = os.path.split(mdfile)
        if opath is None:
            output_dir = fpre
        os.makedirs(opath, exist_ok=True)
        doc = MarkRaw.convert_file(mdfile,opath)
        doc.generate_txt(fpre)
    else:
        doc = MarkTex.convert_file(mdfile,opath)
        doc.generate_tex(fpre)

print(f"[info*]convert finished.")
exit(0)
