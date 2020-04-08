import tkinter as tk

window = tk.Tk()
window.title('t3')
window.geometry('300x300')

l = tk.Label(window, text='0', width=20, height=2, bg='red', fg='black')
l.pack()


def print_select(v):
    l.config(text=v)    


_s = tk.Scale(window, label='try me', from_=0, to=10, orient=tk.VERTICAL,
              length=2000, showvalue=0, tickinterval=2, resolution=0.1, command=print_select)
_s.pack()
window.mainloop()
