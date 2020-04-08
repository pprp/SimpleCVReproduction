import tkinter as tk 

window = tk.Tk()
window.title("t8")
window.geometry('300x300')

# for i in range(3):
#     for j in range(3):
#         tk.Label(window, text=1).grid(row=i, column=j, padx=10, pady=10, ipadx=10, ipady=10)

# tk.Label(window, text='P', fg='red').pack(side='top')    # 上
# tk.Label(window, text='P', fg='red').pack(side='bottom') # 下
# tk.Label(window, text='P', fg='red').pack(side='left')   # 左
# tk.Label(window, text='P', fg='red').pack(side='right')  # 右

tk.Label(window, text='Pl', font=('Arial', 20), ).place(x=50, y=100, anchor='nw')


window.mainloop()