import tkinter as tk

window = tk.Tk()

window.title('t2')

window.geometry('600x600')

#################################
_var_txt = tk.StringVar()
_var_txt.set("initing....")

l = tk.Label(window, bg='black', fg='white', text='select nothing')
l.pack()

_var1 = tk.IntVar()
_var2 = tk.IntVar()

def print_select():
    if _var1.get() == 1 and _var2.get() == 1:
        l.config(text='select both')
    elif _var1.get() == 1:
        l.config(text='select python')
    elif _var2.get() == 1:
        l.config(text='select java')
    else:
        l.config(text="select nothing")

_c1 = tk.Checkbutton(window, text='python', variable=_var1, onvalue=1, offvalue=-1, command=print_select)
_c2 = tk.Checkbutton(window, text='java', variable=_var2, onvalue=1, offvalue=-1, command=print_select)
_c1.pack()
_c2.pack()



#################################

window.mainloop()