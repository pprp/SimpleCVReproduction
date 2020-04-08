# -*- coding: UTF-8 -*-
import tkinter as tk

window = tk.Tk()
window.title('t1')
window.geometry('500x500')

var = tk.StringVar(value="test")
#################################
lb = tk.Label(textvariable=var, bg='yellow', width=20, height=2)
lb.pack()
#################################

on_hit = False
def hit_me():
    global on_hit
    if on_hit == False:
        on_hit = True
        var.set("fucking you")
    else:
        on_hit = False
        var.set('test')
btn = tk.Button(window, text='hit me', width=10, height=2, command=hit_me)
btn.pack()

#################################

ety = tk.Entry(window)
ety.pack()
#################################

txt = tk.Text(window, height=3)
txt.pack()
#################################
var2 = tk.StringVar()
var2.set((1,2,3,4))
lbs = tk.Listbox(window, listvariable=var2)
lbs.insert(1, "first")
lbs.insert(2, "second")
lbs.pack()
#################################
def print_select():
    lb.config(text="vvvvvv?"+var.get())

r1 = tk.Radiobutton(window, text='pig', variable=var, value='pig', command=print_select)
r1.pack()
r2 = tk.Radiobutton(window, text="cat", variable=var, value='cat', command=print_select)
r2.pack()
r3 = tk.Radiobutton(window, text="dog", variable=var, value='dog', command=print_select)
r3.pack()



window.mainloop()