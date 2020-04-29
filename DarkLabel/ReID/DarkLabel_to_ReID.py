import os
import shutil
import cv2
import numpy as np
import glob
import sys
import random
"""[summary]
根据视频和darklabel得到的标注文件
"""

def preprocessVideo(video_path):
    '''
    预处理，将视频变为一帧一帧的图片
    '''
    if not os.path.exists(video_frame_save_path):
        os.mkdir(video_frame_save_path)

    vidcap = cv2.VideoCapture(video_path)
    (cap, frame) = vidcap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    cnt_frame = 0

    while (cap):
        cv2.imwrite(
            os.path.join(video_frame_save_path, "frame_%d.jpg" % (cnt_frame)),
            frame)
        cnt_frame += 1
        print(cnt_frame, end="\r")
        sys.stdout.flush()
        (cap, frame) = vidcap.read()
    vidcap.release()
    return width, height


def postprocess(video_frame_save_path):
    '''
    后处理，删除无用的文件夹
    '''
    if os.path.exists(video_frame_save_path):
        shutil.rmtree(video_frame_save_path)


def extractVideoImgs(frame, video_frame_save_path, coords):
    '''
    抠图
    '''
    x1, y1, x2, y2 = coords
    # get image from save path
    img = cv2.imread(
        os.path.join(video_frame_save_path, "frame_%d.jpg" % (frame)))

    if img is None:
        return None
    # crop
    save_img = img[y1:y2, x1:x2]
    return save_img


def bbox_ious(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * \
                 (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area


def bbox_iou(box1, box2):
    # format box1: x1,y1,x2,y2
    # format box2: a1,b1,a2,b2
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    i_left_top_x = max(a1, x1)
    i_left_top_y = max(b1, y1)

    i_bottom_right_x = min(a2, x2)
    i_bottom_right_y = min(b2, y2)

    intersection = (i_bottom_right_x - i_left_top_x) * (i_bottom_right_y -
                                                        i_left_top_y)

    area_two_box = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1)

    return intersection * 1.0 / (area_two_box - intersection)


def restrictCoords(width, height, x, y):
    x = max(1, x)
    y = max(1, y)
    x = min(x, width)
    y = min(y, height)
    return x, y


if __name__ == "__main__":

    total_cow_num = 0

    root_dir = "./data/videoAndLabel"
    reid_dst_path = "./data/reid"
    done_dir = "./data/done"

    txt_list = glob.glob(os.path.join(root_dir, "*.txt"))
    video_list = glob.glob(os.path.join(root_dir, "*.mp4"))

    for i in range(len(txt_list)):
        txt_path = txt_list[i]
        video_path = video_list[i]

        print("processing:", video_path)

        if not os.path.exists(txt_path):
            continue

        video_name = os.path.basename(video_path).split('.')[0]
        video_frame_save_path = os.path.join(os.path.dirname(video_path),
                                             video_name)

        f_txt = open(txt_path, "r")

        width, height = preprocessVideo(video_path)

        print("done")

        # video_cow_id = video_name + str(total_cow_num)

        for line in f_txt.readlines():
            bboxes = line.split(',')
            ids = []
            frame_id = int(bboxes[0])

            box_list = []

            if frame_id % 30 != 0:
                continue

            num_object = int(bboxes[1])
            for num_obj in range(num_object):
                # obj = 0, 1, 2
                obj_id = bboxes[1 + (num_obj) * 6 + 1]
                obj_x1 = int(bboxes[1 + (num_obj) * 6 + 2])
                obj_y1 = int(bboxes[1 + (num_obj) * 6 + 3])
                obj_x2 = int(bboxes[1 + (num_obj) * 6 + 4])
                obj_y2 = int(bboxes[1 + (num_obj) * 6 + 5])

                box_list.append([obj_x1, obj_y1, obj_x2, obj_y2])
                # process coord
                obj_x1, obj_y1 = restrictCoords(width, height, obj_x1, obj_y1)
                obj_x2, obj_y2 = restrictCoords(width, height, obj_x2, obj_y2)

                specific_object_name = video_name + "_" + obj_id

                # print("%s:%d-%d-%d-%d" %
                #       (specific_object_name, obj_x1, obj_y1, obj_x2, obj_y2),
                #       end='\n')
                # sys.stdout.flush()

                # mkdir for reid dataset
                id_dir = os.path.join(reid_dst_path, specific_object_name)

                if not os.path.exists(id_dir):
                    os.makedirs(id_dir)

                # save pic
                img = extractVideoImgs(frame_id, video_frame_save_path,
                                       (obj_x1, obj_y1, obj_x2, obj_y2))
                print(type(img))

                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    print(specific_object_name + " is empty")
                    continue

                # print(frame_id)
                img = cv2.resize(img, (256, 256))

                normalizedImg = np.zeros((256, 256))
                img = cv2.normalize(img, normalizedImg, 0, 255,
                                    cv2.NORM_MINMAX)

                cv2.imwrite(
                    os.path.join(id_dir, "%s_%d.jpg") %
                    (specific_object_name, frame_id), img)

            max_w = width - 256
            max_h = height - 256

            # 随机选取左上角坐标
            select_x = random.randint(1, max_w)
            select_y = random.randint(1, max_h)
            rand_box = [select_x, select_y, select_x + 256, select_y + 256]

            # 背景图保存位置
            bg_dir = os.path.join(reid_dst_path, "bg")
            if not os.path.exists(bg_dir):
                os.makedirs(bg_dir)

            iou_list = []

            for idx in range(len(box_list)):
                cow_box = box_list[idx]
                iou = bbox_iou(cow_box, rand_box)
                iou_list.append(iou)

            # print("iou list:" , iou_list)

            if np.array(iou_list).all() < 0:
                img = extractVideoImgs(frame_id, video_frame_save_path,
                                       rand_box)
                if img is None:
                    print(specific_object_name + "is empty")
                    continue
                normalizedImg = np.zeros((256, 256))
                img = cv2.normalize(img, normalizedImg, 0, 255,
                                    cv2.NORM_MINMAX)
                cv2.imwrite(
                    os.path.join(bg_dir, "bg_%s_%d.jpg") %
                    (video_name, frame_id), img)

        f_txt.close()
        postprocess(video_frame_save_path)
        shutil.move(video_path, done_dir)
        shutil.move(txt_path, done_dir)