import tkinter as tk

window = tk.Tk()
window.title('t4')
window.geometry('500x300')

_canvs = tk.Canvas(window, bg='green',  height=200, width=500)

imageFile = tk.PhotoImage(file='assets/demo.png')
image = _canvs.create_image(250, 0, anchor='n', image=imageFile)


x0, y0, x1, y1 = 100, 100, 150, 150
line = _canvs.create_line(x0-50, y0-50, x1-50, y1-50)                   # 画直线
oval = _canvs.create_oval(x0+120, y0+50, x1+120, y1 +
                          50, fill='yellow')  # 画圆 用黄色填充
arc = _canvs.create_arc(x0, y0+50, x1, y1+50, start=0,
                        extent=180)      # 画扇形 从0度打开收到180度结束
rect = _canvs.create_rectangle(
    330, 30, 330+20, 30+20)                  # 画矩形正方形

_canvs.pack()


def moveit():
    _canvs.move(rect, 2, 2)
# 移动正方形rect（也可以改成其他图形名字用以移动一起图形、元素），按每次（x=2, y=2）步长进行移动


b = tk.Button(window, text='move item', command=moveit).pack()

window.mainloop()
