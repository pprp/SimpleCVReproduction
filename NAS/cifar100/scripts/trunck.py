import os
import shutil

"""
author:pprp
description： 用于删除空文件夹
"""

path = "exp"

for d in os.listdir(path):
    weight_path = os.path.join(path, d, "weights")
    if not os.path.exists(weight_path):
        print("remove %s" % os.path.join(path, d))
        shutil.rmtree(os.path.join(path, d))
    else:
        if len(os.listdir(weight_path)) == 0:
            print("remove %s" % os.path.join(path, d))
            shutil.rmtree(os.path.join(path, d))
