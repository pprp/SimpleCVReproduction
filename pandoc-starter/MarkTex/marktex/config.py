'''Don't change the basic param'''
import os
'''prog path'''
config_path = os.path.split(__file__)[0]
marktemp_path = os.path.join(config_path,"markenv.tex")

'''tools setting'''
image_download_retry_time = 10
# 在尝试重试次数达到上限后，是否等待手动下载该文件放到目录
# wait_manully_if_all_failed = False
# 在tex文件里添加图片的时候，使用相对路径还是绝对路径
give_rele_path = True

