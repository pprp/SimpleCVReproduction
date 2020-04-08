import tkinter
class MyCanvas:#画布类
    def __init__(self):
        self.status=0#在myButton类的事件处理函数中改变
        self.draw=0#一次操作完成的标志位
        self.canvas=tkinter.Canvas(root,bg='yellow',width=600,height=480)#引用全局变量root
        self.canvas.pack()
        '''画布实例绑定鼠标事件'''
        self.canvas.bind('<ButtonRelease-1>',self.Draw)#鼠标左键释放
        self.canvas.bind('<Button-2>',self.Exit)#鼠标中键按下
        self.canvas.bind('<Button-3>',self.Del)#鼠标右键按下
        self.canvas.bind_all('<Delete>',self.Del)#Del建按下
        self.canvas.bind_all('<KeyPress-d>',self.Del)#键盘D键按下
        self.canvas.bind_all('<KeyPress-e>',self.Exit)#键盘E键按下
    def Draw(self,event):
        if self.draw==0:
            #两点确定一个图形,所以此处获取第一个点坐标
            self.x=event.x#动态增加MyCanvas实例属性
            self.y=event.y#相对于窗口，鼠标光标当前的位置
            self.draw=1
        else:#当点击第二次时,draw标志位改为1进入else判断子句,获取第二个点坐标进行画图
            if self.status==0:
                self.canvas.create_line(self.x,self.y,event.x,event.y)
                self.draw=0
            elif self.status==1:
                self.canvas.create_arc(self.x,self.y,event.x,event.y)
                self.draw=0
            elif self.status==2:
                self.canvas.create_rectangle(self.x,self.y,event.x,event.y)
                self.draw=0
            else:
                self.canvas.create_oval(self.x,self.y,event.x,event.y)
                self.draw=0
    def Del(self,event):
        items = self.canvas.find_all()#找到所有Canvas绘图组件,以元组形式保存
        for item in items:
            self.canvas.delete(item)
    def Exit(self,event):
        root.destroy()#销毁自我,摧毁这个小部件和所有后代小部件
    def SetStatus(self,status):#由MyButton类的按钮事件调用
        self.status=status
class MyLabel:#标签类
    def __init__(self):
        self.text=tkinter.StringVar()#生成标签引用变量(追踪变量）,在myButton类的事件处理函数中改变
        self.text.set('Draw Line')#设置标签变量初始值
        self.label=tkinter.Label(root,textvariable=self.text,fg='red',width=45)#引用全局变量root
        self.label.pack(side='left')
class MyButton:#按钮类
    def __init__(self,type):
        if type==0:
            button=tkinter.Button(root,text='直线',command=self.DrawLine)#实例化绘制直线按钮
        elif type==1:
            button=tkinter.Button(root,text='弧形',command=self.DrawArc)#实例化绘制弧形按钮
        elif type==2:
            button=tkinter.Button(root,text='矩形',command=self.DrawRec)#实例化绘制矩形按钮
        elif type==3:
            button=tkinter.Button(root,text='椭圆',command=self.DrawOval)#实例化绘制椭圆按钮
        button.pack(side='left')
    def DrawLine(self):#绘制直线事件处理函数
        label.text.set('直线')#修改标签文本为直线
        canvas.SetStatus(0)#设置canvas对象标志位为0
    def DrawArc(self):#绘制圆弧/扇形事件处理函数
        label.text.set('弧形')
        canvas.SetStatus(1)
    def DrawRec(self):#绘制矩形事件处理函数
        label.text.set('矩形')
        canvas.SetStatus(2)
    def DrawOval(self):#绘制椭圆函数
        label.text.set('椭圆')
        canvas.SetStatus(3)
if __name__=="__main__":
    root=tkinter.Tk()#实例化窗体对象root
    canvas=MyCanvas()#实例化面板类对象canvas
    label=MyLabel()#实例化文本标签对象label
    #以上的root,canvas,label可被各类的构造方法直接引用
    MyButton(0)
    MyButton(1)
    MyButton(2)
    MyButton(3)
    root.mainloop()

