import tkinter as tk 

window = tk.Tk()
window.title("t5")
window.geometry("300x400")


var = tk.StringVar()
l = tk.Label(window, text=' ', bg='yellow', textvariable=var, width=20, height=3)
l.pack()

def do_job():
    l.config(text="select" + var.get())

menubar = tk.Menu(window)

filemenu = tk.Menu(menubar, tearoff=0)

menubar.add_cascade(label='File', menu=filemenu)

filemenu.add_command(label='new', command=do_job, value='new')
filemenu.add_command(label='delete', command=do_job)
filemenu.add_separator()
filemenu.add_command(label='exit', command=window.quit)

editmenu = tk.Menu(menubar, tearoff=0)

menubar.add_cascade(label='Edit', menu=editmenu)

editmenu.add_command(label='cut', command=do_job)
editmenu.add_command(label='copy', command=do_job)

# 二级菜单
submenu = tk.Menu(editmenu)

editmenu.add_cascade(label='Import', menu=submenu, underline=0)

submenu.add_command(label='pdf', command=do_job)
submenu.add_command(label='png', command=do_job)

window.config(menu=menubar)
window.mainloop()