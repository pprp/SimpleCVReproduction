# -*- coding:utf-8 -*-
from __future__ import division

import glob
import os
import random
import tkinter.messagebox
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askdirectory

from PIL import Image, ImageTk

w0 = 1  # 图片原始宽度
h0 = 1  # 图片原始高度

# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black',
          'Gainsboro', 'FireBrick', 'Salmon', 'SaddleBrown', 'Linen', 'Wheat',
          'Cornsilk', 'GreenYellow', '#6B8E23']

# image sizes for the examples
SIZE = 3101, 1150

# 指定缩放后的图像大小
DEST_SIZE = 1000, 500


def drawCircle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)


class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("Landmark Annotation Tool")
        self.parent.geometry("1000x500")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=TRUE, height=TRUE)

        # initialize global state
        self.imageDir = ''  # 图片所在文件夹
        self.imageList = []

        self.egDir = ''
        self.egList = []

        self.outDir = ''  # 输出文件夹

        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        # TODO 这部分需要改
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # ----------------- GUI 部件 ---------------------
        # dir entry & load
        self.label1 = Label(self.frame, text="ImageDir:")
        self.label1.grid(row=0, column=1, sticky=E+W)

        self.label2 = Label(self.frame, text="SaveDir:")
        self.label2.grid(row=1, column=1, sticky=E+W)

        self.btn1 = Button(self.frame, text="选择图片目录",
                           command=self.get_image_dir)
        self.btn1.grid(row=0, column=2, sticky=E+W)

        self.btn2 = Button(self.frame, text="选择保存目录",
                           command=self.get_save_dir)
        self.btn2.grid(row=1, column=2, sticky=E+W)

        self.lbs_w = Label(self.frame, text='width:')
        self.entry_w = Entry(self.frame)

        self.lbs_w.grid(row=2, column=1, sticky=E+W)
        self.entry_w.grid(row=2, column=2, sticky=E+W)

        self.lbs_h = Label(self.frame, text='height:')
        self.entry_h = Entry(self.frame)

        self.lbs_h.grid(row=3, column=1, sticky=E+W)
        self.entry_h.grid(row=3, column=2, sticky=E+W)

        self.ldBtn = Button(self.frame, text="开始加载", command=self.loadDir)
        self.ldBtn.grid(row=4, column=1, columnspan=2, sticky=N+E+W)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross', bg='red')
        # 鼠标左键点击
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        # 鼠标移动
        self.mainPanel.bind("<B1-Motion>", self.mouseMove)
        # press <Espace> to cancel current bbox
        self.parent.bind("<Escape>", self.cancelBBox)
        # 快捷键
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("a", self.prevImage)  # press 'a' to go backforward
        self.parent.bind("d", self.nextImage)  # press 'd' to go forward
        self.mainPanel.grid(row=0, column=0, rowspan=9, sticky=W+N+S+E)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text='Point Coords:')
        self.lb1.grid(row=5, column=1, columnspan=2, sticky=N+E+W)

        self.listbox = Listbox(self.frame)  # , width=30, height=15)
        self.listbox.grid(row=6, column=1, columnspan=2, sticky=N+S+E+W)

        self.btnDel = Button(self.frame, text='Delete', command=self.delBBox)
        self.btnDel.grid(row=7, column=1, columnspan=2, sticky=S+E+W)
        self.btnClear = Button(
            self.frame, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=8, column=1, columnspan=2, sticky=N+E+W)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=9, column=0, columnspan=3, sticky=E+W+S)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev',
                              width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>',
                              width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(0, weight=30)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(6, weight=1)

        # menu
        self.menubar = Menu(self.parent)
        self.helpmenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='帮助', menu=self.helpmenu)

        self.helpmenu.add_command(label='使用说明', command=self.usage)
        self.helpmenu.add_command(label='关于软件', command=self.about)

        # for debugging
        # self.loadDir()
        self.parent.config(menu=self.menubar)

    def usage(self):
        messagebox.showinfo(
            title='使用说明', message="1. 选择图片所在路径\n2. 选择保存路径\n3.设置保存图片size\n4. 点击开始加载")

    def about(self):
        messagebox.showinfo(title='关于软件', message="作者:pprp 版权所有 请勿商业使用")

    def get_image_dir(self):
        self.imageDir = askdirectory()
        print(self.imageDir)

    def get_save_dir(self):
        self.outDir = askdirectory()
        print(self.outDir)

    def loadDir(self, dbg=False):
        # if not dbg:
            # s = self.entry.get()
            # self.parent.focus()
            # self.category = int(s)
        # else:
            # s = r'./images'
        # print('self.category =%d' % (self.category))

        # self.imageDir = os.path.join(r'./images', '%03d' % (self.category))
        # print(self.imageDir)
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'))
        if len(self.imageList) == 0:
            print('No .jpg images found in the specified dir!')
            messagebox.showwarning(
            title='警告', message="对应图片文件夹中没有jpg图片")
            return
        else:
            print("num=%d" % (len(self.imageList)))

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # set up output dir
        # self.outDir = os.path.join(r'./labels', '%03d' % (self.category))
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)

        # load example bboxes
        self.egDir = os.path.join(r'./Examples', '%03d' % (self.category))

        filelist = glob.glob(os.path.join(self.egDir, '*.jpg'))
        self.tmp = []
        self.egList = []
        random.shuffle(filelist)
        for (i, f) in enumerate(filelist):
            if i == 3:
                break
            im = Image.open(f)
            r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
            new_size = int(r * im.size[0]), int(r * im.size[1])
            self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
            self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
            self.egLabels[i].config(
                image=self.egList[-1], width=SIZE[0], height=SIZE[1])

        self.loadImage()
        print('%d images loaded from %s' % (self.total, self.imageDir))

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        pil_image = Image.open(imagepath)
        # get the size of the image
        # 获取图像的原始大小
        global w0, h0
        w0, h0 = pil_image.size

        # 缩放到指定大小
        pil_image = pil_image.resize(
            (DEST_SIZE[0], DEST_SIZE[1]), Image.ANTIALIAS)

        #pil_image = imgresize(w, h, w_box, h_box, pil_image)
        self.img = pil_image

        self.tkimg = ImageTk.PhotoImage(pil_image)

        self.mainPanel.config(width=max(self.tkimg.width(), 400),
                              height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt = int(line.strip())
                        continue
                    print(line)
                    tmp = [(t.strip()) for t in line.split()]

                    print("********************")
                    print(DEST_SIZE)
                    print("tmp[0,1]===%.2f, %.2f" %
                          (float(tmp[0]), float(tmp[1])))
                    print("********************")

                    self.bboxList.append(tuple(tmp))
                    tmp[0] = float(tmp[0])
                    tmp[1] = float(tmp[1])

                    tx0 = int(tmp[0]*DEST_SIZE[0])
                    ty0 = int(tmp[1]*DEST_SIZE[1])

    def saveImage(self):
        # print "-----1--self.bboxList---------"
        print(self.bboxList)
        # print "-----2--self.bboxList---------"

        with open(self.labelfilename, 'w') as f:
            f.write('%d\n' % len(self.bboxList))
            for bbox in self.bboxList:
                f.write(' '.join(map(str, bbox)) + '\n')
        print('Image No. %d saved' % (self.cur))

    # 鼠标事件
    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1 = self.STATE['x']
            y1 = self.STATE['y']

            x1 = x1/DEST_SIZE[0]
            y1 = y1/DEST_SIZE[1]

            self.bboxList.append((x1, y1))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, 'loc:(%.2f, %.2f)' % (x1, y1))
            self.listbox.itemconfig(
                len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

            drawCircle(self.mainPanel, self.STATE['x'], self.STATE['y'], 5, fill=COLORS[(
                len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text='x: %.2f, y: %.2f' % (
            event.x/DEST_SIZE[0], event.y/DEST_SIZE[1]))  # 鼠标移动时显示当前位置的坐标
        # 如果有图像的话，当移动鼠标时，展示十字线用来定位
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(
                0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(
                event.x, 0, event.x, self.tkimg.height(), width=2)

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

    def imgresize(self, w, h, w_box, h_box, pil_image):
        '''
        resize a pil_image object so it will fit into
        a box of size w_box times h_box, but retain aspect ratio
        '''
        f1 = 1.0*w_box/w  # 1.0 forces float division in Python2
        f2 = 1.0*h_box/h
        factor = min([f1, f2])
        # print(f1, f2, factor) # test
        # use best down-sizing filter
        width = int(w*factor)
        height = int(h*factor)
        return pil_image.resize((width, height), Image.ANTIALIAS)


if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.mainloop()
