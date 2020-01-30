import os
import shutil

import cv2
import random
import matplotlib.pyplot as plt


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)


file_contents = "./dpjsave/406_augment.txt"
jpg_path = "./dpjsave/406_augment.jpg"

img = cv2.imread(jpg_path)

f = open(file_contents,"r")

height, width, _ = img.shape

f_c = f.readlines()

for line in f_c:
    clss, xc, yc, w, h = line.split()
    xc, yc, w, h = float(xc), float(yc), float(w), float(h)

    xc *= width
    yc *= height
    w *= width
    h *= height

    half_w, half_h = w // 2, h // 2
    x1, y1 = int(xc - half_w), int(yc - half_h)
    x2, y2 = int(xc + half_w), int(yc + half_h)

    c = [x1,y1,x2,y2]

    plot_one_box(c, img)

cv2.imshow("dimtarget", img)
cv2.waitKey(0)
