import cv2
import os
import shutil
import tqdm
import sys

root_path = r"I:\Dataset\VideoAnnotation"

def print_flush(str):
    print(str, end='\r')
    sys.stdout.flush()

def genXML(xml_dir, outname, bboxes, width, height):
    xml_file = open((xml_dir + '/' + outname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + outname + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + 'cow' + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(x1) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(y1) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(x2) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(y2) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')


def gen_empty_xml(xml_dir, outname, width, height):
    xml_file = open((xml_dir + '/' + outname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + outname + '.png' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('</annotation>')


def getJPG(src_video_file, tmp_video_frame_save_dir):
    # gen jpg from video
    cap = cv2.VideoCapture(src_video_file)
    if not os.path.exists(tmp_video_frame_save_dir):
        os.makedirs(tmp_video_frame_save_dir)

    frame_cnt = 0
    isrun, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]

    while (isrun):
        save_name = append_name + "_" + str(frame_cnt) + ".jpg"
        cv2.imwrite(os.path.join(tmp_video_frame_save_dir, save_name), frame)
        frame_cnt += 1
        print_flush("Extracting frame :%d" % frame_cnt)
        isrun, frame = cap.read()
    return width, height

def delTmpFrame(tmp_video_frame_save_dir):
    if os.path.exists(tmp_video_frame_save_dir):
        shutil.rmtree(tmp_video_frame_save_dir)
    print('delete %s success!' % tmp_video_frame_save_dir)

def assign_jpgAndAnnot(src_annot_file, dst_annot_dir, dst_jpg_dir, tmp_video_frame_save_dir, width, height):
    # get coords from annotations files
    txt_file = open(src_annot_file, "r")

    content = txt_file.readlines()

    for line in content:
        item = line[:-1]
        items = item.split(',')
        frame_id, num_of_cow = items[0], items[1]
        print_flush("Assign jpg and annotion : %s" % frame_id)

        bboxes = []

        for i in range(int(num_of_cow)):
            obj_id = items[1 + i * 6 + 1]
            obj_x1, obj_y1 = int(items[1 + i * 6 + 2]), int(items[1 + i * 6 + 3])
            obj_x2, obj_y2 = int(items[1 + i * 6 + 4]), int(items[1 + i * 6 + 5])
            # preprocess the coords
            obj_x1 = max(1, obj_x1)
            obj_y1 = max(1, obj_y1)
            obj_x2 = min(width, obj_x2)
            obj_y2 = min(height, obj_y2)
            bboxes.append([obj_x1, obj_y1, obj_x2, obj_y2])

        genXML(dst_annot_dir, append_name + "_" + str(frame_id), bboxes, width,
            height)
        shutil.copy(
            os.path.join(tmp_video_frame_save_dir,
                        append_name + "_" + str(frame_id) + ".jpg"),
            os.path.join(dst_jpg_dir, append_name + "_" + str(frame_id) + ".jpg"))

    txt_file.close()


if __name__ == "__main__":
    append_names = ["cutout%d" % i for i in range(19, 66)]

    for append_name in append_names:
        print("processing",append_name)
        src_video_file = os.path.join(root_path, append_name + ".mp4")

        if not os.path.exists(src_video_file):
            continue

        src_annot_file = os.path.join(root_path, append_name + "_gt.txt")

        dst_annot_dir = os.path.join(root_path, "Annotations")
        dst_jpg_dir = os.path.join(root_path, "JPEGImages")

        tmp_video_frame_save_dir = os.path.join(root_path, append_name)

        width, height = getJPG(src_video_file, tmp_video_frame_save_dir)

        assign_jpgAndAnnot(src_annot_file, dst_annot_dir, dst_jpg_dir, tmp_video_frame_save_dir, width, height)

        delTmpFrame(tmp_video_frame_save_dir)
