import tkinter as tk 

window = tk.Tk()
window.title('t6')
window.geometry('500x500')

tk.Label(window, bg='red', width=30, height=5).pack()

frame = tk.Frame(window)
frame.pack()

f_a = tk.Frame(frame)
f_b = tk.Frame(frame)

f_a.pack(side='right')
f_b.pack(side='left')

tk.Label(f_a, text='????', bg='yellow').pack()
tk.Label(f_a, text='????', bg='yellow').pack()

tk.Label(f_b, text='!!!!', bg='blue').pack()
tk.Label(f_b, text='!!!!', bg='blue').pack()

window.mainloop()