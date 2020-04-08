# -*- coding: UTF-8 -*-
import tkinter as tk

window = tk.Tk()
window.title('t1')
window.geometry('500x500')

var = tk.StringVar(value="test")

lb = tk.Label(textvar=var, bg='yellow', width=20, height=2)
lb.pack()


window.mainloop()