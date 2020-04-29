import os
'''
gt.txt:
---------
frame(从1开始计), id, box(left top w, h),ignore=1(不忽略), class=1(从1开始),覆盖=1), 
1,1,1363,569,103,241,1,1,0.86014
2,1,1362,568,103,241,1,1,0.86173
3,1,1362,568,103,241,1,1,0.86173
4,1,1362,568,103,241,1,1,0.86173

cutout24_gt.txt
---
frame(从0开始计), 数量, id(从0开始), box(x1,y1,x2,y2), class=null
0,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
1,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
2,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
'''


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # y = torch.zeros_like(x) if isinstance(x,
    #                                       torch.Tensor) else np.zeros_like(x)
    y = [0, 0, 0, 0]

    y[0] = (x[0] + x[2]) / 2
    y[1] = (x[1] + x[3]) / 2
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y

def process_darklabel(video_label_path, mot_label_path):
    f = open(video_label_path, "r")
    f_o = open(mot_label_path, "w")

    contents = f.readlines()

    for line in contents:
        line = line[:-1]
        num_list = [num for num in line.split(',')]

        frame_id = int(num_list[0]) + 1
        total_num = int(num_list[1])

        base = 2

        for i in range(total_num):

            print(base, base + i * 6, base + i * 6 + 4)

            _id = int(num_list[base + i * 6]) + 1
            _box_x1 = int(num_list[base + i * 6 + 1])
            _box_y1 = int(num_list[base + i * 6 + 2])
            _box_x2 = int(num_list[base + i * 6 + 3])
            _box_y2 = int(num_list[base + i * 6 + 4])

            y = xyxy2xywh([_box_x1, _box_y1, _box_x2, _box_y2])

            write_line = "%d,%d,%d,%d,%d,%d,1,1,1\n" % (frame_id, _id, y[0],
                                                        y[1], y[2], y[3])

            f_o.write(write_line)

    f.close()
    f_o.close()

if __name__ == "__main__":
    root_dir = "./data/videosample"

    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)

        video_path = os.path.join(full_path, item+".mp4")
        video_label_path = os.path.join(full_path, item + "_gt.txt")
        mot_label_path = os.path.join(full_path, "gt.txt")
        process_darklabel(video_label_path, mot_label_path)
