import tkinter as tk
from tkinter import messagebox

window = tk.Tk()
window.title("t7")
window.geometry('300x300')

def hit_me():
    messagebox.showinfo(title='information', message='你干嘛点我？？傻逼')
    messagebox.showwarning(title='warning', message='你再点我试试，傻逼')
    messagebox.showerror(title='error', message='nmsl')

btn = tk.Button(window, text='hit me', command=hit_me)

btn.pack()


window.mainloop()