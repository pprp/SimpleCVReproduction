# 高等程序设计语言

## 写在最前面的话

下面这些文字，以及每章最开始的引言都来源于最初书写这些知识简介的学长学姐们，初闻不知曲中意，再听已是曲终人，经过这一年的学习，我也对这些知识有了属于我自己的理解，但我始终还是希望你在看到这份资料时，对这些筚路蓝缕的学长学姐们保留里应存在的一些敬意，体会那份他们所寄托在这上面的情感：

以下正文：

由衷地为你开始了你的第一门编程语言的学习感到开心，在国外，以Java为起手作为第一门高级程序设计语言的大学并不多，但这并不意味着Java不好，相反，他们都是以仰视的态度来看待这门语言，认为它是大项目的代名词，因为Java这门语言，是Android应用，金融服务器，网站，大数据Hadoop的一些基础语言，是企业级使用最多的语言。它又是一种纯面向对象的语言，你可能现在不清楚这个词是什么意思，但随着学习的深入你会发现面向对象开发技术的优雅与强大。一开始就接触这种强大的语言，比Python复杂，却比C++省心，这是对你的挑战，更是对你的收获。
但是，注意，这门课的名称叫做高级程序设计，并不叫Java，它的目的是将你带入编程世界，不同编程语言之间总是殊途同归，掌握了一门语言后，你会发现一通百通，所以不要囿于一门语言的边界，重在体会编程的思想，把它当作工具，开始你对计算机世界的探索。
我们希望，通过智库的努力，能够帮助现在的你不要再像当时我们探索时那么迷茫，但更希望你们不要因为智库带来的安逸，失去主动探索的能力。**去突破我们，突破课本的边界吧，这个世界，需要你我一起去探索未知**，希望走在我们铺的小路上的你，能为后来人踏出新的大道！
Welcome!

# 第一章

## 1.1 引言

在学习编程之前，我们首先就要对计算机的硬件部分建立简单的了解，在明白计算机的构成之后，我们再在他的基础上来学习软件，最后就是Java的一些基本特点，作为这个专业的初学者在这门课中对这章知识不需要深究其原理，记住常识知识即可，更重要的是以这章知识为景区入口的大地图，概览风光，心怀期待，剩下的交给以后的日子用脚步去亲自丈量体验！

## 1.2 要点
[![QQ图片20190727122342.png](https://i.loli.net/2019/07/27/5d3bd1db3c04952913.png)](https://i.loli.net/2019/07/27/5d3bd1db3c04952913.png)
### 1.2.1 硬件组成

计算机，顾名思义，最开始它的用途设想是用来计算数据，从这个角度来想：我们需要键盘来输入数据，用来计算的电路（控制器和运算器，现代计算机里包装在一起称CPU），显示屏（IO）和保存中间结果的部分（主存储器）。如果我想保存更多的历史结果和数据怎么办？于是加入了断了电也能用的大容量存储器（辅助存储器）。这就是计算机的组成部分。这样的设计是有原因和思路的。

### 1.2.2计算机的软件组织结构

软件是用来帮助人方便地使用计算机硬件的，高级计算机的软件分为专门与硬件交互，为上层各种软件提供服务的操作系统层和功能各异的应用软件层。简单来说，就是软件有和计算机硬件打交道的，有和人打交道的，还有在之间起连接作用的。这样的分层可以帮助应用软件的开发更为便捷。

### 1.2.3 JAVA

Java是什么？简而言之，一句话：写程序的。我们通过java来创造其他程序并且保证其运行。而且不只是Java，我们常用的程序语言们，例如：C++，Python等，都是同样的作用。

众所周知，Java是一门出现时间较为短暂的程序语言。相较于早期的那些对人类极不友好的晦涩的机器语言和汇编语言来说，Java具有我们能看的懂的自然语言单词，是对人类很友好的高级语言。那么我们如何将现如今的Java转换成那些机器可以识别的1和0呢？

### 1.2.4 编译

你在编辑器里写的Java代码叫Source Code（源代码），后缀名为.java。 Java Compiler（Java编译器）负责将你的源代码统一转化成一种中间形式的代码，叫bytecode（字节码）后缀名为.class，Java为了保证让你写的代码可以不受CPU类别不同的限制在哪里都能运行于是加入了Java虚拟机，然后再由Java虚拟机里的专门的字节码解释器（Bytecode interpreter）和字节码编译器（Bytecode compiler）转换成不同机器的机器码。

### 1.2.5 程序结构 !

终于，到了正式让你认识第一个程序的时候了！
[![87ea1062-5464-42f1-827f-4975f2be264e.png](https://i.loli.net/2019/07/27/5d3c0b83b737798730.png)](https://i.loli.net/2019/07/27/5d3c0b83b737798730.png)


### 1.2.6那个绿色反斜杠后面的东西是什么？

注释是写程序时为了帮助自己和其他编程人员理解的内容，里面可以写任何内容，因为编译时编译器会忽略其内容直接编译执行代码，所以不会影响机器执行。提倡写注释，在软件项目中写好关键注释会给你加分哦！
用//表示这一行都为注释内容
用/* 和 */符号把任意行的内容作为注释。

### 1.2.7 标识符？

标识符事实上就是我们在程序里写的那些单词，由数字，字母下划线和$构成。它不能由数字开头，而且大小写敏感（即大小写有区别）。

他们只能用来表述那些特定作用，不能再做他用。

下面这张图就包含了绝大多数Java保留的关键字，大家只要记住它就可以啦！


[![下载 (1).jpg](https://i.loli.net/2019/07/27/5d3c0bfa8c1af16278.jpg)](https://i.loli.net/2019/07/27/5d3c0bfa8c1af16278.jpg)

### 1.2.8 如果出现错误怎么办？

我们把程序的错误大致分三类：
(1)	语法错误(compile-time errors 编译时错误)
编写时语法不准确，命名不守规则等，这些都是编译器可以发现的错误，写时报的错。
(2)	运行时错误(run-time errors)
语法没错误但是出现除零啊等算数逻辑错误。
(3)	逻辑错误(logical errors)
没有以上的明显的错误，就是程序设计的逻辑有问题出的错误。

遇到错误时，先认真排查，再仔细观察逻辑结构，就能获得世界的真实!!

## **奇妙而美丽的例题！**


1.下列关于计算机系统和Java编程语言的说法，正确的是（C   ）。

A.	字节码程序是计算机硬件能够直接执行的程序。	

B.	在程序中书写注释增加了代码的长度，降低了代码的运行时间。	

C.	Java语言是面向对象的编程语言，所有的main方法都属于某一个类。	

D.	在计算机系统中，信息以二进制形式或者字符串形式存储。	

2.下列选项中符合Java命名规则的标识符是（   BD ）。

A.	4volunteer	B.	Abs$c

C.	Employee-123	D.	_Student

3.下面哪些是Java中合法的标识符。A

A.  MAX_SIZE   B.  2ndLevel  C. coin#value  D. hook&ladder

4.编译Java 源程序文件产生的字节码文件的扩展名为  B

A.  java       B.  class       C.  html      D.  exe

5.关于Java程序的说法，哪些是正确的？

A．	Java程序不能直接被Windows XP操作系统加载运行。

B．	“.java”为后缀名的源文件直接被Java虚拟机加载运行。

C．	.class为后缀名的文件是可以直接运行的二进制可执行文件。

D．	Eclipse是运行Java程序的必需环境。

E．	Java程序的运行环境能够自动的完成常规的内存管理，不用显式的释放内存。

6.下面哪些是Java中合法的命名变量的标识符？

A．	continue 

B．	intValue

C．	123Sample

D．	my-apple

E．	_eclipse

 7．下列关于计算机系统和Java编程语言的说法，正确的是（C）

A．	计算机是由硬件、操作系统和软件组成，操作系统是缺一不可的组成部分。

B．	 Java语言编写的程序源代码可以不需要编译直接在硬件上运行。

C．	 在程序中书写注释不会影响程序的执行，可以多写一些详细的注释。

D．	Java的集成开发环境（IDE），如Eclipse，是开发Java语言必需的软件工具。

8．下列选项中符合Java命名规则的标识符是（D）

A．	2japro    B.  &Class    C.  const    D.  _123




# 第二章.数据类型和表达式

## 2.1引言

如果将一个完整的软件项目比作现实世界里的高楼大厦，其中最基础的部分就是数据类型和表达式，这二者好比大楼的砖瓦。一道程序的效率高低往往是由选取的变量类型和表达式所决定的。从熟悉各种变量和表达式开始，一窥计算机与软件的魅力。

## 2.2 Hello World！！！

每个程序员都是从输出一句"Hello World!"开始的，但是不同的语言有不同的输出语句，Java中我们输出它的方式是 System.out.println ("Hello World!"); 。
不要小瞧了这行简单的代码，通过它可以认识到Java的很多特点。首先这是一句经典的函数调用，其次它涉及到最经典的字符串变量。下面将为大家介绍Java的函数调用和字符串类型变量。

### 2.2.1 Hello World——String类型

可以这样讲，字符串（String）类型是最经典的数据类型。String类型的成功是多方面的，它不仅可以很容易的拆分和拼接，还可以与其他数据类型很容易的转化。所以熟悉String类型的使用是很有必要的。

(1)	字符拼接

String类型的拼接使用“+”即可。但是要注意与算数加法区分。
举个例子：（来自PPT的经典永不过时老例题hhhh）

System.out.println ("24 and 45 concatenated: " + 24 + 45);
System.out.println ("24 and 45 added: " + (24 + 45));

输出是：
24 and 45 concatenated: 2445
24 and 45 added: 69

前者把24和25当作字符串处理，而第二者则是先做了算术加法。所以小伙伴们在进行字符串的同时如果还需要进行算术运算，记得把算术加法用（）括起来哦(*^_^*)

(2)	转义字符

 [![u=1856383793110435479&fm=26&gp=0.jpg](https://i.loli.net/2019/07/27/5d3c0c721d98514152.jpg)](https://i.loli.net/2019/07/27/5d3c0c721d98514152.jpg)


很多时候我们希望字符串的输出格式规范一些，这时我们可以用一系列转义	字符来实现我们的需求。当Java看到字符串中的“\”符号时，它会知道这是	要对字符串的格式进行操作了，所以“\”被称为转义符。但是当这样我们想简单的输出“\”就需要用“\\”来表达了。

## 2.3 八大金刚——基本数据类型

在刚才hello world 当中，我们已经认识了String类型的变量，但String并不是基本数据类型，Java的基本数据类型只有8个，他们分别是：byte, short, int, long,float, double,char,boolean。

### 2.3.1 数值类型

(1)	整数

整数类型的表示相对简单，就是将日常中使用的十进制转化为二进制数。
byte, short, int, long都属于整数类型变量，但是它们的表达数据的能力有些区别。
byte：占8位bit（1字节） 表达范围：-128~127
short：占16位bit（2字节） 表达范围：-32768~32767
int：占32位bit（4字节） 表达范围：-2,147,483,648~2,147,483,647
long：占64位bit（8字节） 表达范围：-9 x 10^18~9 x 10^18
因为需要表示正负数，所以每种类型的第一位数用0表示正数，1表示负数。除此以外，正数比负数少一个，因为有0需要表示。所以一个占Mbit的数，它的表示范围是-2^(M-1)~2^(M-1)-1。

这次我们举个西瓜:int类型占据了32bit，所以他的表述范围就在-2^(32-1)~2^(32-1)-1即-2,147,483,648~2,147,483,647位。

(2)	浮点数

浮点数的表示和整数类型大不相同，它更接近于我们使用的科学计数法。

vfloat, double都是浮点数类型，它们的表达能力比正数类型大许多。float是32位，也称为单精度浮点数。double是64位，也称为双精度浮点数。关于浮点数的使用和表示，今后的课程《计算机组成原理》有详细的分析，这里不深入解读。

(3)	数据类型转换

不同的数据类型之间可以相互转换，表达能力小的变成大的自然没什么问题，但是表达能力大的变成小的会出问题，要谨慎使用。

### 2.3.2.1 错与对——Boolen

Boolen类型只有两种值：ture 和 false，而计算机的起源，便是用开和关的电信号来进行逻辑运算的，由此看来，Boolean还颇有点返璞归真的意思。我们常在哪里使用它呢？

举个桃子这次：假若我们需要判断一个学长到底是不是帅气而正直的那只，进而决定要不要找这个小哥哥疯狂沙雕，我们就可以用Boolean来表示我们的结果，假如是ture，那我们就可以疯狂快乐，如果是false，我们就需要对他提高警惕了哟(*^_^*)

PS：（山软智库里的小哥哥小姐姐经过官方认证，全部都是ture，所以还在等什么？干嘛不来找我们玩？叉腰气鼓鼓！）

[![output_1564216547.gif](https://i.loli.net/2019/07/27/5d3c0cf232e8e92989.gif)](https://i.loli.net/2019/07/27/5d3c0cf232e8e92989.gif)

### 2.3.2.2 我是String的弟弟——char

char类型只能保存一位字符：比如‘a’、‘b’。在字符串处理时经常使用。说到这，就会有人想：兀那小贼，明明是个字符，还想妄图混上数值类型的贼船？

但事实上并不是的，Java字符存储使用Unicode编码，每个字符对应2个字节（16位）的二进制数字，Java要根据这串数字在字库中找到对应字符。因此int a = ‘a’;这个语句可以得到字符‘a’的Unicode编码。也就是说，char类型的数值转换其实是Unicode编码被转换成不同数值类型。

这里有点生涩，大家可以简单理解为我们给每个字符都编了序号，然后当使用Char类型存储字符的时候，实质上可以说是存储了它的编号。所以char类型其实是他的序号被转换成不同数值类型。

## 2.4 表达式

(1) 变量赋值

变量赋值使用的符号是“=”，它将右边的值赋给左边。这个值要符合左边数据类型的规范。另外如果变量前有final关键字的约束，那么这个变量一经赋值其值便无法被第二次赋值。

举个例题：

下面赋值语句中正确的是（A）

A. double d=5.3e12;	B. float f=11.1;

C. int i=0.0;		

这道题值得大家记住它。其中的A.B选项很容易迷惑大家，A是浮点数的科学计数法，B没有f标记，这是大家容易忽视的知识点。

(2) 牵涉数据类型转换的运算

Java中表达式的使用和现实世界中的加减乘除没有分别，运算优先级也么什么不同，但是由于多种数据类型的加入，其中必然牵扯到变量的转换，因此我们需要了解其中的一些特性。

需要注意的有以下几点：

1.	两个数据类型相同的数作运算，结果和他们的数据类型相同。因此整数类型的结果只保留整数部分。
2.	两个数据类型不同的数作运算，先将表达能力小的转换成大的，再作运算，结果是表达能力大的那种类型。

结合下面的例子来理解：
```java
public static void main(String[] args){
		int count = 12;
		double sum = 490.27;
		System.out.println(sum / count);
	}		
```	
分析：先将count转成double类型，再计算sum/count。输出的结果是double类型，输出：40.85583333333333。

3.	在运算中显式的类型转换，数据类型与转换类型相同，其精度也于转换类型相同。
结合下面的例子来理解：
```java 
public static void main(String[] args){
		int total = 50;
		int count = 6;
		float result = (float) total / count;
		System.out.println(result);
	}
```
分析：total和count都是int类型，如果不强制类型转换，那么输出的结果将是8，整数之后的部分会丢失。而强制转换之后小数点之后的部分被保留下来，且精度为float类型，输出：8.333333。
注意：这一部分内容只是看讲解可能印象并不深刻，建议大家设计几个小例子，亲自动手试一试。

经典例题：
如下Java语句 double x=2.0; int y=4; x/=++y;  执行后，x的值是（C）
A.  0.5    B.  2.5    C.  0.4   D.  2.0
分析：x/=++y是表达式的简写方式，将它展开写成：x=x/(y+1)，x是double类型，y是int类型，结果应该是double类型，结果等于2/5=2.5。

## 举个大例子！

1.设x是float型变量，y 是double型变量，a是int型变量，b是long型变量，c 是char型变量，则表达式x+y*a/x+b/y+c 的值为___C___类型。

A . int  B. long  C. double  D. char

2.下列表达式正确的是（   ）

A.	Byte b = 254;	B.	float a = 1.0;

C.	double d = 0.999f;	D.	char c = -12;

3．	下面赋值语句中正确的是（A）

A.	 double d=5.3e12;      B.   float f=11.1;

C.	 int i=0.0;            D.   Double oD=3;

4．	下列关于Java语言中String和char的说法，正确的是（C）

A.	 String是Java定义的一种基本数据类型。

B.	 String是以“\0”结尾的char类型的数组char[]。

C.	 使用equals()方法比较两个String是否内容一样（即字符串中的各个字符都
一样）。

D.	 Char类型在Java语言里面存储的是ASCII码。

5.下列赋值语句，代码可以正确执行的是：E

A.	 boolean flag=0;

B.	 char c=97.0;

C.	 int a=’a’;

D.	 double d=’a’;

E.	 short b=30000;

6．	下面赋值语句中正确的是（A）

A.	 double d=5.3e12;      B.   float f=11.1;

C.	 int i=0.0;            D.   Double oD=3;

7.选出正确的表达式 AD

A.	double a=1.0; 

B.	Double a=new Float(1.0); 

C.	byte a = 1340; 

D.	Byte a = 120;





# 第三章.使用类和对象

## 3.1 引言 

本书前四章都是对概念的基本介绍，学的时候可能一头雾水，但是在多多编程后心中许多疑惑就会解开，真实的编程会告诉你为什么需要这样那样。多多动手熟悉eclipse，熟悉每一种学过的典型方法，就会发现这些枯燥的概念其中的奥妙之处。

忍不住在这里再添一点，说实话第四章类和对象里有大量的枯燥，冗长的概念，希望大家尽可能的去理解，多多动手。一个不会写代码的程序员和残废有什么区别？大雾

## 3.2 一个看似是句子的名词——对象引用变量

### 3.2.1 对象引用变量的声明和初始化

在Java中，我们如何使用一个变量呢？正如同那个经典的冰箱放大象的例子，使用变量也只需要三步：

声明变量；初始化变量；使用

[![QQ图片20190727163817.png](https://i.loli.net/2019/07/27/5d3c0d8271e1477315.png)](https://i.loli.net/2019/07/27/5d3c0d8271e1477315.png)

- 声明
  
  经过声明的变量开始并没有存放任何数据（没有初始化）。必须将变量初始化才能使用。当然也可以将其设置为null，表示不指向任何对象，空

- 初始化
  
  对象初始化后，可用对象名.方法(参数)来引用该对象的方法。没有参数也要保留括号以标志方法。其中某些方法还会有返回值，比如name.length()就是返回该String对象的长度，需要有一个int对象来存储这个返回值，即count= name.length();

上图表格中出现的new运算符相当于调用类的一种特殊方法：构造方法，该方法名与类名相同，会初始化一个新的对象将参数传递进去以后返回该对象引用变量的地址。

而事实上String虽然不是基本类型，但是因为常用所以可以直接和基本类型一样声明和初始化合并为一步操作，如下面的例子：
String name=new String(“H”);

### 3.1.2 别名

由于基本类型变量名代表的是该类型的值，而对象引用变量名代表的是地址，所以在出现classA=classB的赋值语句时，会有不同的结果。

[![ce9a1372-3d76-437a-9e9c-c1f6b40123fd.png](https://i.loli.net/2019/07/27/5d3c0ddb3854c35365.png)](https://i.loli.net/2019/07/27/5d3c0ddb3854c35365.png)
 
可以看到，基本类型在赋值以后只是将num2内存单元中的数据修改成了num1中的，二者还是不同的内存单元，但是对象引用变量则是在赋值后二者指向了同一个内存单元，而name2原本指向的那个对象因为没有了引用而用不会被使用，该对象称为垃圾，会被java的垃圾回收机制处理释放该资源。

这两种类型赋值时的本质是一样的，在classA=classB时都是classB中内容替换classA中内容，只不过对象引用变量中内容是某个对象的引用（也就是某个对象的地址），所以才会出现不同。
对于上面name1和name2，它们引用相同对象而变量名不同所以互为对方别名。


举个浅显一点的例子：假设现在有一个旅馆，里面有两间客房，第一个例子就相当于两间客房里住了一模一样的客人（把第一间房的客人克隆到第二间房间）。而第二个例子就相当于复制了第二间房的门牌号，并且把复制好的第二间房的门牌号也贴在第一间房门上，这样两个房间号指向同一个房间。而那个没有门牌号的房间就被清空回收另作他用。

## 3.2 包以及常用的类和方法

Java语言有一个标准类库支持，可以根据需要直接使用。类库的类被划分成若干个包，每个包都有自己特有的功能。当用户想要使用某个功能的时候，可以在程序最开头使用import声明将该包中的某个类引入程序，这个类就变成了可以使用的类。
如果用户想要使用一个包中的许多类，可以直接将整个包引入。如：
```java 
import java.util.Scanner;	//引入java.util 包中的Scanner类
import java.util.*;		//引入java.util 包中所有类
```
这里再明晰一下类的概念。在前面的学习中，我们接触到了基本类型和String类型，它们都有自己对应的方法。而实际上在java中有很多其他的或者来自标准类库或者人为定义的一些类，如pen 或者dog。这些将在第四章展开讨论，这里只需要明晰类的概念即可。
关于类库/包，只需要了解概念，实际应用中借助编译器就可以使用。当然如果对于常用的包比较熟悉更好。
注意，java.lang包是最基本的包，可视为语言的基本扩展，所以会被自动导入程序，包中任何类如String，System都可以直接使用而不需要import。

### 3.2.1 Random类——骰子摇一摇！

Random 类在java.util 包中，通过使用它我们能产生一个伪随机数
使用示例：
```java 
1)	Random rand=new Random();//rand是个能随机数的“种子”/生成器
2)	int num1=rand.nextInt();//num1是rand随机产生的一个int值
3)	num1=rand.nextInt(10);//num1是rand随机产生的一个0-9区间的int值
4)	num1=rand.nextInt(10)-1;//num1是rand随机产生的一个-1-8区间的int值
5)	float num2=rand.nextFloat();//num2是rand随机产生的一个[0.0,1.0)浮点数
```
### 3.2.2 Math类——数据算一算

这个类中有着大量基本数学函数，定义在java.lang 包中。Math类所有方法都是静态方法，可以直接使用 类名.方法(参数) 调用而不需要实例化类的对象。第6章详细讨论静态方法。
Math类也同上只需了解不用把所有方法都记下来。具体使用方法如下：
int a=Math.abs(-3);	//a的值为-3的绝对值

### 3.2.3 NumberFormat类——数据格式化

有getCurrencyInstance获取当前货币格式和getPercentInstance获取百分率格式两种方法。以第一种为例进行演示：
```java
NumberFormat fmt1=NumberFormat. getCurrencyInstance();//fmt1设置为当前地区货币值格式对象
System.out.println(“价格 : ”+fmt1.format(11));//将11以fmt1的格式输出
```
结果：价格 : ¥11

我们在这里为大家列出了很多常用的类和方法，但如果同学们在编程时还对自己需要使用的类和方法不很明确，这个时候我们的建议是面向搜索引擎编程，什么不会查一查哈哈哈哈。

## 课后练一练

1、下列变量定义中，错误的是

A、int x = 3; B、float f;d;

C、String s = "Hello!"D、boolen b = ture ;

2、下列变量定义错误的是D

 A) int a; B) double B = 4.5; C) Boolean B = true; float f = 9.8;

3、下列关于包（package）的描述，正确的是（D）

A．	包（package）是Java中描述操作系统对多个源代码文件组织的一种方式。

B．	import语句将所对应的Java源文件拷贝到此处执行。

C．	包（package）是Eclipse组织Java项目特有的一种方式。

D．	定义在同一个包（package）内的类可以不经过import而直接相互使用。



# 第四章 想用什么找不到？？？那就自己写一个——编写类

在最开始编写类时，我们需要了解对象和类的关系、区别；之后需要熟悉类的结构（数据、方法）、基础知识（返回值、参数列表等），最后需要能够根据类的特点自己设计它所包含的变量、方法。

那让我们预习基本概念，然后开始自己编写一个类吧！

## 4.1 基本概念

类与对象：类是一类相似事物的总和，而对象则是具体存在的一个事物，比如人是个类，而你就是一个对象；又或者学生是一个类，而你是学生这个类的一个对象。
对象：对象是类的一个实体，是有状态和行为的。

对象的状态：由内部的属性（变量）表示，行为由函数实现，例如：学生张三是学生类的一个对象，它有一些状态，比如学习的状态，这可以用一个变量来表示（例如用boolean类型的变量study表示是否学习，true表示正在学习，false为玩耍状态）。

对象的行为：以学生张三为例，它存在上课这个行为，所以可以定义一个函数Learn（）来表示它上课的行为，每次调用这个函数就表示张三进行了上课这个行为。

注意：对象的行为可能会改变对象的状态，上面的例子中当张三执行上课这个函数时它就会改变张三的状态，改变成学习状态。

换个说法，类就像是一张蓝图，我们可以根据这张蓝图修出具有同样结构的房子（将类实例化为对象）。之后再给这个房子增添属于它自己的风格和艺术气息（实例化对象后具体做改变：例如重写方法等）。

## 4.2 举个栗子写个类！

### 4.2.1 编写类的方法：

(1)声明类包含的各种变量
(2)编写类的各种方法，方法中可能会修改类的变量。类的方法包括构造方法和常规方法。写方法时需要考虑方法的返回值、参数等。

那么，什么又是构造方法和实例数据呢？

构造方法的方法名称跟类名称相同、方法不能有返回值（void返回值也不可以），创建一个对象时会调用它来初始化一个对象（一般是用来初始化对象的一些变量值等）。不必为每个类都创建构造方法，因为每个类都有一个不带参数的默认构造方法。

而实例数据则是类中的属性（变量），每次创建一个对象时会为这个对象的实例数据分配内存空间，这就是为什么同一个类的不同对象都具有自己的状态。变量的声明位置定义了这个变量的可见范围，变量的可见范围确定了变量在程序中已被引用的区域。当变量声明在类的级别（不是在某个方法内部声明），类内的所有方法都能引用这个变量。

### 4.2.2 写个骰子当例子

(1) 分析
以Die类（骰子）为例：因为Die不是Java类库的预定义类，所以需要我们自己进行类的编写，需要编写类的数据和方法：

①类的数据声明：分析骰子的各种状态，它有最大面值和当前面值，所以需要两个变量来表示这两个状态。

②类的方法的声明：骰子具有的行为有旋转、设置当前面值、获取当前面值，所以需要三个方法来分别实现这三个行为。除了类本身的各种行为以外，还可能会声明一些方法用于信息的输出，例如下面要看的方法toString().

Die类包含两个变量（实例数据）：整型常量MAX来表示骰子的最大面值,整型变量faceValue来表示骰子当前面的数值，这两个实例数据的声明在类的级别，所以Die类内的所有的方法都可以引用它们。

Die类包含构造方法和普通方法。它的构造方法是方法名和类名Die相同的方法，也就是方法Die()，它实现的是把当前面的数值初始化成1.当使用new运算符来创建一个Die对象的时候，系统会调用Die类的构造方法Die()完成facevalue变量初始化为1。
常规方法包括骰子自身的一些行为和用于输出信息的方法。其中骰子自身的行为包括roll(),setFaceValue(),getFaceValue()三个，这跟之前分析的骰子的行为是一致的。

Die类跟之前例子中的类没有什么不同，唯一的区别在于Die是我们自己编写的，而其他的是Java标准类库提供的。

代码实现见下图（来自书上和ppt上的超级经典老例子）

[![hjlk.png](https://i.loli.net/2019/07/27/5d3c0eefc270296266.png)](https://i.loli.net/2019/07/27/5d3c0eefc270296266.png)
### 4.2.3 UML，描述面向对象程序设计的符号体系。

UML：统一建模语言，最流行的一种描述面向对象程序设计的符号体系。

UML类图：描述类的结构及其类间的关系的UML图。

用法：UML类图中每个类用矩形表示，其中由三部分组成：类名、属性（数据）、操作（方法）。

类间关系：以类A，类B为例

①“A uses B”:A使用B的方法

②“A has B”：类A内部有B类型的变量

③“A is B”：A类属于B类，例如人类属于生物这个类。

https://www.cnblogs.com/pangjianxin/p/7877868.html

这个博客可以很好的帮助我们理解UML图的更多具体信息

### 4.2.4 封装，提高安全性。

封装：对象必须是独立运行的，内部变量必须经过自己的方法修改，类之外的代码难于甚至无法访问和修改在类内部声明的变量，这种特性叫做封装。而加入我们想要获得对象的数据，必须通过对象的方法，而不能直接获得。

### 4.2.5 private和public——数据可见性

Java中可见性修饰符包括private和public。

Public：修饰的方法和变量是公开可见的，所有类都可见（可以引用），UML图中在变量前面加“+”表示public可见性。

Private：修饰的方法和变量则只能在类的内部可见，类外不可见（之前的Die类的MAX,
faceValue变量都是private修饰，所以只在Die类内可见），因为Private修饰的方法只在类的内部可见，所以它一般辅助类内的其他方法工作，也称为支持方法，UML图中在变量前面加“-”表示private可见性。

而假设我们要获得相关数据，我们就可以使用访问器，而假若我们要修改它，这就需要另一个方法。

举个例子：若变量名是Weight，则我们若想访问该数据，则应该使用getHeight()，而假若我们要对该数据做修改，则就要用到setHeight（）。

## 4.3 方法 

### 4.3.1 方法声明

函数（方法）包括函数头、函数体两部分，所以函数的编写也需要按这两部分进行编写。

(1)函数头

①返回类型：函数要返回给调用位置的数据，若没有返回值则写void作为返回值类型。
②参数：
实际参数：一次方法调用中，实际传递给方法的参数是实际参数（实参，也称为方法的变元）。
形式参数（形参）：是方法声明时的参数，方法调用时形参的初始值由调用该方法时传递的实参赋予。
③参数列表：调用这个函数时需要给出的数据，给出的参数列表（数据的类型、排列顺序等）需要跟函数的声明时的参数列表完全一致，若该方法没有参数也需要在方法名后面写一个空括号。

(2)函数体

①返回语句：存在返回值时需要有返回语句return。当执行返回语句时控制立即返回到调用该方法的位置并继续往下执行。返回语句由return和后续的表达式组成，表达式确定了要返回的值，并且类型和函数头规定的返回值类型一致。一般一个方法只有一个return语句，写在方法的最后一行。

没有例子不能说题的我

[![12-130Q1220955916.jpg](https://i.loli.net/2019/07/27/5d3c106f1366413562.jpg)](https://i.loli.net/2019/07/27/5d3c106f1366413562.jpg)

方法包含一个方法头和一个方法体。下面是一个方法的所有部分：

修饰符：修饰符，这是可选的，告诉编译器如何调用该方法。定义了该方法的访问类型。

返回值类型 ：方法可能会返回值。returnValueType 是方法返回值的数据类型。有些方法执行所需的操作，但没有返回值。在这种情况下，returnValueType 是关键字void。

方法名：是方法的实际名称。方法名和参数表共同构成方法签名。

参数类型：参数像是一个占位符。当方法被调用时，传递值给参数。这个值被称为实参或变量。

参数列表是指方法的参数类型、顺序和参数的个数。参数是可选的，方法可以不包含任何参数。
方法体：方法体包含具体的语句，定义该方法的功能。

通过这个例子相信大家可以很好的了解方法的组成和结构，剩下的java教程，我们也会迅速在近期推出，届时敬请期待哦！

## 课后练一练

1．	下列关于Java中类的构造方法的描述，正确的是（B）

A． 构造方法的返回类型为void

B. 可以定义一个类而在代码中不写构造方法。

C. 在同一个类中定义的重载构造方法不可以相互调用。

D. 子类不允许调用父类的构造方法。

2．	下列关于Java类中方法的定义，正确的是（D）

A.	若代码执行到return语句，则将当前值返回，而且继续执行return语句后面的
语句。

B.	只需要对使用基本数据类型定义的属性使用getter和setter，体现类的封装性。

C.	方法的返回值只能是基本数据类型。

D.	在同一个类中定义的方法，允许方法名称相同而形参列表不同，并且返回值数据类型也不同。

3、请完善下列程序，能够让程序正确编译运行，而且得到所示的正确输出。
 
 Vehicle类建模了所有交通工具，Motor类建模了摩托车，摩托车不能够载客，只有1个驾驶员；Bus类建模了公交车，公交车可以有多个驾驶员和乘客，总承载人数为（驾驶员+乘客）。（12分）
```java
public abstract class Vehicle {
		private int drivers;// 驾驶员人数
		public int getDrivers() {
			return drivers;
		}
		public abstract int getLoads();// 得到载人的总数，包括驾驶员和乘客
		//请完善构造器（Constructor），如果不需要构造器，可以不写（2分）
```

4.在Java中，用Package语句说明一个包时，该包的层次结构必须是：A

A.	与文件目录的层次相同

B.	与文件的结构相同

C.	与文件类型相同 

D.	与文件大小相同 

5．	下列关于Java中类的构造方法的描述，正确的是（B）

A． 构造方法的返回类型为void

B. 可以定义一个类而在代码中不写构造方法。


#   第五章.条件判断和循环
##  5.1 布尔表达式
  **条件语句和循环语句可用于控制程序的执行流程**。
 - 条件语句有时也叫选择语句，它允许选择下一步要执行哪一条语句，在Java中，条件语句主要有if，f-else和switch语句。
 - 循环语句可以使程序多次执行某些语句，Java中循环语句主要有while,do-while和for语句。
### 5.1.1等式运算符和关系运算符
  - 等式运算符：==（等于）！=（不等于）
  - 关系运算符：<(小于） <=(小于等于） >(大于） >=(大于等于)

### 5.1.2逻辑运算符
  - 逻辑非：**！**      *a为真，则！a为假*
  - 逻辑与：**&&**      *a、b同为真才是真*
  - 逻辑或：**||**	   *a、b只要有一个是真就为真*

优先级：逻辑非>逻辑与>逻辑或
短路问题：运算符**&&**和**||**具有短路性，简单说就是在判断真假时，如果左边的操作数已经可以确定整个运算的结果，那么右边的操作数就不会再参与运算。例如当&&左边是false了，那么无论右边是什么这个表达式都是假，所以右边不回参与运算，发生短路；而当||左边是真，那么整个表达式肯定是真，右边的操作数不会发生运算，发生短路。

>举个例子:  
``` javascript
	int i=0,j=0;
	if(++i>0||++j>0)
	System.out.print(i);
	System.out.print(j);
```
输出：1 0

## 5.2 if语句
if(条件){
	
	执行此处语句
}
>栗子：
``` javascript
if(5>1){
	System.out.println("5>1");
}
```
输出 5>1


### 5.2.1 if-else语句
if(条件){
	
	执行A
}else{
	
	执行B
	}

>栗子：
``` javascript
	if(a>1){
		System.out.println("a>1");
	}else{
		System.out.println("a<=1");
	}
```
当a=5时，输出a>1
当a=1时，输出a<=1

### 5.2.2使用语句块
直白的讲，当我们的程序要执行多条语句时，把这些语句用**{}**括起来，这就是一个语句块。

if(条件){
	
	执行A
	执行B
	执行C
	…………
}

大括号内的就是语句块，当条件成立时，语句块里的语句会全部执行。

### 5.2.3if语句的嵌套
一条if语句中嵌入另一条if语句，这种情况叫if的嵌套。
在if的嵌套中值得我们关注的是if与else的配对情况：**在一个if嵌套语句中，else子句与它前面最近的且未匹配的if语句相匹配**

如:
``` javascript
	if(code=='R'){
		if(height<=20)
			System.out.println("S");
		else
			System.out.println("B");
	}
```
上述代码else和第二个if匹配。

但是，我们可以用括号界定else子句属于哪个if语句。

如：
``` javascript
	if(code=='R'){
		if(height<=20)
		System.out.println("S");
	}
	else{
	System.out.println("B");
	}
```

此处else和第一个if配对.

## 5.3数据比较
### 5.3.1 浮点数比较

根据==运算符的意义，只有当两个浮点数的二进制数位都相等时，这俩数的值才相等，可这样很难达到精确的相等，所以应该尽量少用这个方法。
**判断两个浮点数相等的一个较好的方法时计算两个数的差值并将差和某个误差标准比较。**

### 5.3.2字符比较

在Java中字符以Unicode字符集为基础，在字符集定义了所有可能用到的字符的顺序，因为字符‘a’在‘b’前面，所以‘a’小于‘b’.

常用顺序：……  数字  ……  大写字母 …… 小写字母  ……
### 5.3.3对象比较

字符串不是基本数据类型，而是对象。

比较字符串大小可以按照字典顺序：从两个字符串的第一个字母开始比较，第一个字母大的字符串也大，第一个字母相同时，比较下一个字母，以此类推。如果一个字符串是另一个字符串的前缀时，短字符比长字符小。
>  horse<horsefly


相关方法：compareTo（）

判断字符串是否相等：

**==：判断的是两个字符串的地址是否相等**

** equals（）：判断两个字符串的内容是否相等**

## 5.4while语句
 
 while语句是循环语句，重复执行while语句块里的操作，直到它的条件变为false。
 
 ``` javascript
	int count=1;
	while(count<=5){
		System.out.print(count);
		count++;
	}
 ```
 >输出：12345
 
 ### 5.4.1无限循环
 while的判断条件永远是真时，循环将一直执行下去或者直到这个程序被终止。我们必须谨慎设计代码来避免无限循环。
 
 >look!这有个坑！！！
 ``` javascript
	double num=1.0;
	while(num!=0.0)
	num=num-0.1;
 ```
 实际上，这个循环是无限的，因为num的值不会精确的等于0.0.数值是二进制表示的，系统内部发生的极小的误差都会影响两个浮点数的比较。
 
 ### 5.4.2 循环嵌套
 while里面套while，此时要有认真读术的亚子！
 ### 5.4.3 break和continue语句
 
 这两个可以控制程序的执行，可以终止循环或者跳出循环。
 
 **break：跳转到控制当前执行流程语句之后执行。**
 
 **continue：跳出本次循环，再次计算循环控制条件，若其值是真，则再次执行循环体。**
 
 >送你栗子!
 
	 while（a<10）{
	 
	 语句A…………
	 
	 break;（直接跳出while循环，执行while后面的语句C）
	 
	 语句B………… 
	 }
	 
	 语句C…………
	 

 
	 while(a<10){

	语句A …………
	
	 continue;（跳出本次循环，不再执行语句B就会到while判断上去，若条件为真就再执行语句A…………
	
	语句B …………
	
	 }
 
	 语句C…………
 

# 第六章.深入的条件判断和循环
## 6.1 switch语句

switch语句也是选择结构，使用多个if语句可以构造出同样的逻辑，但是使用switch语句可以使代码的可读性更强。
switch语句先计算一个表达式的值，然后将该值和几个可能的case子句取值进行匹配，每种取值都有与它关联的执行语句。

>栗子：
``` javascript
	switch（idChar）{
		case 'A':
			aCount++;
			break;
		case 'B':
			bCount++;
			break;
		default:
			System.out.println("End");
 ```
 
 switch语句将一个指定的字符或整型值分别与若干个case子句中的值进行匹配。上述代码中，首先计算表达式（一个简
 单的字符变量），然后程序开始将表达式和下面的case语句匹配，从第一个出现的case开始，若匹配则执行case A中相
 应的语句，若不和第一个匹配就继续处理第二个case语句，以此类推。若都没有case与之匹配，就执行default中的语句。
 
 值得注意的是其中的**break**，break是跳出选择结构，例如当idChar=='A'时，执行aCount++,然后break，跳出这
 个选择结构，执行选择结构之后的程序；若没有这个break，程序会顺序执行：aCount++,再执行case B 中的语句：
 bCount++………………。
 
 **break语句经常用在switch语句的每个case子句结束处。**
 
 **switch语句中开始的表达式运算结果必须是char,byte,short或int类型。具体的说不能为boolean,float和string类型，而且每一个case子句中的表达式必须为常量，不能是变量或者表达式。**
 
 **switch语句隐含的布尔条件是基于等式的，不能进行其他的关系运算判断**
 
 ## 6.2 条件运算符
条件运算符从某些方面来讲可以作为if-else语句的缩写，但它的可读性不如if-else。

条件运算符是三元运算符，因为它需要三个操作数，通常书写成？：，例如：(a>b)?a:b;问号前是一个布尔条件，其后被：分开的是两个表达式。

如果布尔条件为true，则结果取冒号前边的，否则取冒号后边的表达式的值。在上面的例子(a>b)?a:b中，就是取最大值的运算。

**条件运算符基于比尔表达式计算出两个可能值中的一个值。**

## 6.3 do语句

do语句和while语句很相似，也是重复执行循环体中的语句，直到条件变为false，但do循环会至少执行一次，while循环最少一次也不执行。

>while循环：

while(布尔条件){
	
	循环体语句
	
}

布尔条件为真时执行循环体语句。

>do循环：

do{
	
	循环体语句
	
}while(布尔条件)

先执行一次循环体，再判断布尔条件，只有条件为真时再返回while重复执行循环体语句。


## 6.4 for语句
for语句是又一种循环语句，特别适用于循环执行前已知具体循环次数的情况。

形式：

for(初始化;布尔表达式;增量){
	
	循环体语句
	
}

>栗子:
``` javascript
	for(int i=0;i<100;i++){
		System.out.println(i);
	}
 ```
上面的程序是换行输出1~99.

>如果倒序输出99~0，应该：

``` javascript
	for(int i=99;i>=0;i--){
		System.out.println(i);
	}
 ```

for循环可以看做while循环的变形：
>
``` javascript
	int i=50;
	while(i<77){
		System.out.println(i);
		i++;
	}
 ```
 相当于：
 ``` javascript
 	for(int i=50;i<77;i++){
		System.out.println(i);
	}
  ```
  上面两个例子都是输出50~76.
  
  ### 6.4.2 循环的比较
其实各种循环的功能都是等价的，具体使用哪种循环看具体情况。

当已知循环的条件时，建议用while循环或do循环(至少执行一次用do循环)

当已知循环的次数时，建议用for循环。

## 课后题：
>典例1:写一个九九乘法表。
 ``` javascript
class Test01{
	public static void main(String[]args) {
		for(int i=1;i<=9;i++) {
			for(int j=1;j<=i;j++) {
				System.out.print(i+"*"+j+"="+i*j);
			}
			System.out.println();
		}
	}
}
  ```
>典例2：读程序写结果。

 ``` javascript
import java.util.Scanner;

public class GradeReport
{
   public static void main (String[] args)
   {
      int grade, category;

      Scanner scan = new Scanner (System.in);

      System.out.print ("Enter a numeric grade (0 to 100): ");
      grade = scan.nextInt();

      category = grade / 10;

      System.out.print ("That grade is ");
switch (category)
      {
         case 10:
            System.out.println ("a perfect score. Well done.");
            break;
         case 9:
            System.out.println ("well above average. Excellent.");
            break;
         case 8:
            System.out.println ("above average. Nice job.");
            break;
         case 7:
            System.out.println ("average.");
            break;
         case 6:
            System.out.println ("below average. You should see the");
            System.out.println ("instructor to clarify the material "
                                + "presented in class.");
            break;
         default:
            System.out.println ("not passing.");
      }
   }
}

  ```
>输入：91

Enter a numeric grade (0 to 100): 91

That grade is well above average. Excellent.
  
>典例3：课本例题精读。

 ``` javascript
//本例题灵活运用了判断语句和类与对象的概念，是全书的第一道上台阶的例题，建议精读。
public class CoinFlip
{
   public static void main (String[] args)
   {
      Coin myCoin = new Coin();//创建Coin对象myCoin

      myCoin.flip();//抛硬币一次

      System.out.println (myCoin);//调用toString方法，返回结果

      if (myCoin.isHeads())//判断是否是头朝上
         System.out.println ("You win.");
      else
         System.out.println ("Better luck next time.");
   }
}

public class Coin
{
   private final int HEADS = 0;
   private final int TAILS = 1;

   private int face;
//构造函数
   public Coin ()
   {
      flip();//调用filp()，模拟随机抛硬币过程，给face随机赋值为0或1
   }
   public void flip ()
   {
      face = (int) (Math.random() * 2);//随机取0或1
   }

  public boolean isHeads ()//判断是否是正面朝上
   {
      return (face == HEADS);
   }
   public String toString()//重写toString函数，若头朝上返回Heads，否则返回Tails
   {
      String faceName;

      if (face == HEADS)
         faceName = "Heads";
      else
         faceName = "Tails";

      return faceName;
   }
}
  ```

  # 第七章.面向对象设计
## 7.1 引言
对象说白了也是一种数据结构(对数据的管理模式)，将数据和数据的行为放到了一起。在内存上，对象就是一个内存块，存放了相关的数据集合。
## 7.2 明确类和对象
### 7.2.1 类的概念
**类**是模版，或者图纸，系统根据类的定义来造出对象。我们要造一个汽车，怎么样造?类就是这个图纸，规定了汽车的详细信息，然后根据图纸将汽车造出来。类是用于描述同一类型的对象的一个抽象概念，类中定义了这一类对象所应具有的共同的属性、方法。
### 7.2.2 对象的概念
**对象**就是类的实例。就像上面所说的，根据图纸制造出来的汽车就是一个对象。对象的属性，指的是该对象拥有的变量，以及该对象课调用的方法，这些都是类里面设定好的。不同对象就是根据同一个图纸，加以不同的调整制造出来的。
## 7.3 静态类成员
### 7.3.1 静态变量
静态变量是由类的所有成员共享的，静态变量的声明为**static int count =0**
### 7.3.2 静态方法
**静态方法**也成为了类方法，是通过类名而不是对象调用的，且静态方法只能引用静态变量和局部变量
### 7.3.3 各种不同的变量和方法
下面我们来理清一下静态变量、实例变量、局部变量这三种变量，另外区分静态方法和非静态方法
![关系示例](https://i.loli.net/2019/08/22/zZyOeIvlVjFQBHk.png)
## 7.4 类间关系
### 7.4.1 依赖关系
通常一个类的方法需要调用另一个类的方法，这样就建立了类间的“使用”关系。可以认为一个类必须依靠另一个类。
### 7.4.2 聚合关系
一个聚合对象由其他的对象组成，既是把其他对象的引用作为自己的实例数据，形成一种“有”关系。例如，汽车作为一个对象“有”底盘，而底盘也是一个对象。
### 7.4.2 this引用
指的是该类本身的一个对象（当前实例）。在构造方法中，this的本质就是创建好的对象的地址，由于在构造方法调用前，对象已经创建。因此，在构造方法中也可以使用this代表“当前对象” 。
## 7.5 接口
 **接口是抽象方法的集合，因而不能被实例化**，抽象方法是没有实现的方法，没有代码体。 一个类可以实现多个接口，一个类实现了接口，必须实现接口中所有的方法，并且这些方法只能是public的。因此，接口就是比抽象类更加抽象，可以更加规范的对子类进行约束。下面则是一个接口和一个类实现接口的实例
 ![接口实例](https://i.loli.net/2019/08/22/xUfkWBNlMJcyGZP.png)
 ![实现接口实例](https://i.loli.net/2019/08/22/XHetdlx1CWKfwVa.png)
## 7.6 枚举类型
枚举类型可以理解成自己定义的数据类型，而枚举类型的值，是具有该枚举类型的静态变量。在枚举类型的定义中，可以增加属性和方法。
![QQ浏览器截图20190822135644.png](https://i.loli.net/2019/08/22/nz3GEdYAPODwNQ5.png)
![QQ浏览器截图20190822135656.png](https://i.loli.net/2019/08/22/u89xgarYXmKenS2.png)
## 7.8 方法设计
### 7.8.1 方法分解
**可以将对象提供的一个复杂服务，分解为由多个方法支持的简单服务**，可以理解为把方法的步骤细分出来，每一个步骤对应一个方法。
### 7.8.2 方法参数的传递方式
**将对象传递给方法时，形参和实参相互称为对方的别名**。当我们定义一个函数时**void add（int a, int b）**，这里的a和b就是形参。当我们调用该方法时，**add（1, 2）**这里的1和2就是实参。
![无标题.png](https://i.loli.net/2019/08/22/B4CkazonXG65e3Y.png)
### 7.8.3 方法重载
同名的方法，不同的参数列表。**多个重载方法可以有参数个数、类型及参数顺序来区分**，用于对不同类型的数据需要执行类似的方法。
![QQ浏览器截图20190822141200.png](https://i.loli.net/2019/08/22/ZKLjQI1zbXB26Uo.png)
## 7.9 相关例题
下面给大家列出往年的相关试题
![第七章选择题一.png](https://i.loli.net/2019/09/18/qHMiwp4egRnoxA7.png)
![第七章选择题二.png](https://i.loli.net/2019/09/18/7rNv6qkc2bHdIZj.png)
![第七章选择题三.png](https://i.loli.net/2019/09/18/SdDn4QLWEON71rF.png)
# 第八章 数组
## 8.1 引言 
数组是想同类型数据的有序集合。数组的索引总是从0开始的。**具有N个值的数组索引为0到N-1**
![一维数组的结构](https://i.loli.net/2019/08/22/Lwy5GrNYtQROiMx.png)
## 8.1 声明和使用数组
要想使用一个数组，就要先进行声明和初始化。**声明**就是确定数组名字和数据类型，**初始化**就是给数组开辟内存空间。
### 8.1.1 数组声明方式：
数组的声明，顾名思义就是告诉系统：我要创建一个数组。其中，声明的内容就是数组名和数据类型。
![数组有两种声明方式](https://i.loli.net/2019/08/21/PatTYmcizE6yZDe.png)

### 8.1.2 数组初始化方式
数组的初始化方式有三种：
+ （1）动态初始化：直接在定义数组的同时就为数组元素分配空间并赋值
+ （2）静态初始化：动态初始化数组，先分配空间，再给数组元素赋值
+ （3）默认初始化：若不给数组元素赋值，数组元组将为默认值
![QQ浏览器截图20190821215910.png](https://i.loli.net/2019/08/21/7A6jfa4HrPc35Y2.png)
### 8.1.3 数组作为参数
在面向对象的世界里，数组也是**对象**。整个数组可以作为一个参数传递给方法，此时方法的形参称为原始数组的别名。
![数组作为参数传递原理](https://i.loli.net/2019/08/22/WUQg5aXEpfdC2BJ.png)
## 8.2 对象数组
数组除了可以储存基本类型数据，还可以将对象引用作为元素保存。最直接的例子就是装字符串的数组，其结构如下
![对象数组的结构](https://i.loli.net/2019/08/22/snuzT3pbAwr4gUj.png)
## 8.3 命令行实参
命令行实参存储在String对象的数组中，并将传给main方法。具体统发如下图
![QQ浏览器截图20190822101819.png](https://i.loli.net/2019/08/22/fxjXiMcqED6ygls.png)
## 8.4 可变类型参数表
**Java可以定义参数可变的方法**实际上，该方法能接受任意个数的参数，并将参数自动存入数组，以便在方法中进行处理。
![可变类型参数表](https://i.loli.net/2019/08/22/sh4eOcb3m8Pl2MH.png)
## 8.5 二维数组
我们以上说的都是一位数组，而二维数组是有二位的值，可以看成由行和列构成的表，该表由多个一位数组组成。
![命令行实参](https://i.loli.net/2019/08/21/aHqc3TEbk9nJwGZ.png)
## 8.6 多维数组
在了解过一维和二维数组之后，相信多位数组就不难理解了。不过需要知道的是**在面向对象的系统中，很少使用高于二维的多维数组**
## 8.7 相关例题
下面给大家列出往年的相关试题
![第八章选择题一.png](https://i.loli.net/2019/09/18/TpxOeVz7BnREYhg.png)
![第八章选择题二.png](https://i.loli.net/2019/09/18/1ELqp9ngHBVsGzo.png)
![第八章读程序题.png](https://i.loli.net/2019/09/18/O3iNLtGxU1DKd6H.png)

# 第九章.继承
## 9.1 引言
继承是面向对象的三大特性之一。细胞分化大家都很熟悉了，继承跟分化类似。**继承就是从现有类派生出新类的过程**，可以说是父类属性的细分和扩展。
![QQ浏览器截图20190822142311.png](https://i.loli.net/2019/08/22/xkvhHAMtOXE68oU.png)
继承在父类和子类之间建立一种“是”关系。如图，无论动物还是植物都是属于生物。
### 9.1.1 protected访问控制符
**protected可见性提供了允许继承的最大可能封装性**，Java中4种访问控制符分别为**private**、**default（默认，即什么都不加）**、**protected**、**public**，这四个访问控制符的权限是一次递增的，它们说明了面向对象的封装性。
![QQ浏览器截图20190822151745.png](https://i.loli.net/2019/08/22/dDPquO6RoGVs7Kj.png)
### 9.1.2 super的用法
super是直接父类对象的引用。可以通过super来访问父类中被子类覆盖的方法或属性。若是构造方法的第一行代码没有显式的调用super()或者this();那么Java默认都会调用super(),含义是调用父类的无参数构造方法。这里的super()可以省略。
![QQ浏览器截图20190822145936.png](https://i.loli.net/2019/08/22/ci7zNUkhI145e8K.png)
### 9.1.3 Java的继承机制
Java的继承机制为**单继承**，既是一个子类只能有一个父类。
## 9.2 重写方法
子类方法可以重写（重新定义）它所继承的父类方法。方法的重写需要注意：父类方法和子类方法在**方法名、形参列表**上相同。详见9.1.2的图。
## 9.3 类层次结构
### 9.3.1 类层次结构详解
**一个类的子类还能被其他类所继承，称为其他类的父类，由此建立起一种链式的类层次结构**，就像引言的图中，动物类是生物类的子类，同时是哺乳动物类的父类。继承机制具有传递性，一个被继承的特性可能来自于父类，也可能来自若干层以上的祖先类。因此我们要合理地将类的共同性保持在尽可能高的类层次上。
### 9.3.2 Object类
**所有的Java类都直接或间接的由Object类派生**，因此Java程序的每一个类都继承toString方法和equals方法。如果在类的声明中未使用extends关键字指明其父类，则默认继承Object类。-
## 9.4 抽象类
只要含有**抽象方法**的类就是抽象类。抽象类可以含有普通方法（必须实现）。 抽象类不能被实例化，抽象类代表一种概念，子类将基于这种概念来定义方法，可以说抽象类就是用来被继承的。由抽象类派生的子类必须重写所有父类的抽象方法，否则该子类仍然是抽象类。
![QQ浏览器截图20190822144220.png](https://i.loli.net/2019/08/22/wJCM5ouDhy1N8zs.png)
继承的概念可以应用到接口，以便由一个接口派生另一个接口，因此接口可以理解为更简单的抽象类，接口之间是可以继承的。
## 9.5 可见性
父类的私有成员也被子类继承，虽然不能以成员名直接访问这些私有成员，但可以间接访问。此时父类提供相应的get/set方法来访问相关属性，这些方法通常是public修饰的，以提供对属性的赋值与读取操作。
## 9.6 相关例题
下面给大家列出往年的相关试题
![第九章选择题二.png](https://i.loli.net/2019/09/18/XNp9GCHkTLncA4I.png)
![第九章选择题三.png](https://i.loli.net/2019/09/18/QeTmkRfKPqY9uyr.png)
![第九章选择题一.png](https://i.loli.net/2019/09/18/YW2QJrnUvfEkpMy.png)
## **接下来是一道写程序题**
![写程序题（接口）.png](https://i.loli.net/2019/09/18/6uylBwxIq5G1HvR.png)
![写程序题（父类）.png](https://i.loli.net/2019/09/18/TiqwI2kdmXAFt64.png)
![写程序题（Car类）.png](https://i.loli.net/2019/09/18/IR3u1TLHSWGzvDb.png)
![写程序题（Truck类）.png](https://i.loli.net/2019/09/18/zMU2HVXhfJ9OcaP.png)
![main方法.png](https://i.loli.net/2019/09/18/bsaIwQjPeGOpHdA.png)


# 第11章.异常
学到这里，相信不少小机灵鬼们也上网看过专业程序员们写的代码了，这时我们发现在一些项目里，程序员们写的代码和我们写的长得有点不太一样啊。为什么呢？因为专业的程序员们会熟练地处理异常。
那什么是异常呢？异常就是程序中出现的问题或非正常情况。针对一些常常出现的异常，Java有相应的异常对象。


## 11.1异常处理
Java程序中发生的问题可能产生异常或者错误。

一个异常是一个定义非正常情况或错误的对象，由程序或运行时环境抛出，可以根据需要进行捕捉和处理。

一个错误类似于异常，不同的是错误代表不可恢复的问题并且必须捕捉处理。

Java预定义了一组程序执行中可能发生的异常和错误。

**错误和异常都是对象，代表非正常情况或无效处理。**

> 容易引起异常抛出的问题：

- 除数是0的除法
- 数组索引越界
- 找不到指定的文件
- 不能正常完成被请求的I/O操作
- 使用了空引用
- 执行的操作违反了某种安全规则
- 等等……


## 11.2 未捕捉的异常
如果程序不处理异常，程序就不会被正常执行，并且在输出结果中描述在何处发生了什么异常。

``` javascript
	public static void main(String [] args){
		int a=10,b=0;
		System.out.println(a/b);
		System.out.println("End!");
	}
```
> 输出：

``` javascript
	Exception in thread "main" java.lang.ArithmeticException:/ by zero
	
	at zero.main(Zero.java:17)
	}
```

在上述程序中，0做了除数，却没有处理异常，所以当异常发生时程序就会结束执行，并打印有关异常的具体信息。注意最后一条输出End！不会执行，因为在执行它之前就发生了异常。

在输出中，第一行信息表明抛出什么异常，并提供了抛出该异常的原因。其他行是调用堆栈跟踪信息，指明何处发生的异常。

## 11.3 try-catch-finally语句
try-catch语句用于标志可能抛出异常的语句块，try可以有多个相关联的catch子句，每个catch子句称为一个异常处理器。
在其后还可以加finally子句，表示无论异常是否发生或者捕捉处理，都将执行finally子句。

> 举例
>
``` javascript
	try
	{
		System.out.println(10/0);
	}
	catch
	{
		System.out.println("zero can be devided!");
	}
	finally
	{
		System.out.println("End!");
	}
```

> 输出：
``` javascript
	zero can be devided!
	End!
```

finally子句不是必须的，可以写也可以不写，它是无论有无异常，异常无论是否被捕捉处理都会执行的语句。因此常用finally保证一定执行某段资源。

一般的，try-catch-finally语句中，

try语句中不发生异常，就跳过catch语句执行finally语句

try语句中发生了异常，就先执行相应异常的catch语句，再执行finally语句。

## 11.3 异常的传递
如果一个异常的发生处没有进行对该异常的捕捉和处理，控制将立即返回产生该异常的方法的上一级调用方法并在该处进行相应异常的捕捉处理，如果上一级还是没有捕捉和处理，就继续向上返，直到异常被捕捉处理或者返回到了main方法，这时将终止程序的执行并产生异常信息。

> 来，栗子！
``` javascript
	public static void main(String [] args)
	{
		System.out.println("main start!");
		try
		{ 
			f1();
		}
		catch(ArithmeticExeption exeption)
		{
			System.out.println("0 can not be devided!");
		}
		finally
		{
			System.out.println("End");
		}
	}
	void f1()
	{
		System.out.println("f1 start!");
		System.out.println(10/0);
	}
```

>输出：
``` javascript
	main start!
	f1 start!
	0 can not be devided!
	End
```

## 11.4 异常类层次结构

除了Java预定义的异常，我们也可以自己定义异常：从Exeption类或者它的后代类派生一个新类来定义。

> 栗子还要吗
>
``` javascript
	import java.util.Scanner;
	public class CreatExeptionS
	{
		public static void main(String [] args)
		{
			final int MIN=25,MAx=40;
			Scanner scan=new Scanner(System.in);
			OutOfRangeException problem=new OutOfRangeExeption("Input value is out of range.")
			System.out.print("Enter a number between 25 and 40!");
			int value=scan.nextInt();
			if(value<MIN||value>MAX)
			{
				throw problem;
			}
			System.out.println("END!");
		}
	}
public class OutOfRangeExeption extends Exeption
{
	OutOfRangeExeption(String message)
	{
		super(message);
	}
}
```

> 输出
> 
> Enter a number between 25 and 40!   69
> 
> Exeption in thread "main" OutOfRangeExeption:
> 
> Input value is out of range.
> 
> at CreatingExeptions.main(CreatingExceptions.java:20)


### 11.4.2  可检测异常和不可检测异常
异常分为可检测和不可检测异常，

可检测异常必须由一个方法捕捉或者在方法定义声明头的相关throws子句中列出来。

不可检测异常不需要使用throws子句。

java中唯一不可检测的异常是RuntimeExeption类的对象或者后代类对象。

>习题 读程序。

``` javascript
	public class Propagation
{
  static public void main (String[] args)//main函数，程序的入口。
   {
      ExceptionScope demo = new ExceptionScope();

      System.out.println("Program beginning.");//执行
      demo.level1();//跳到level1方法，调用level1()
      System.out.println("Program ending.");//执行
   }
}
public class ExceptionScope
{
   public void level1()
   {
      System.out.println("Level 1 beginning.");//执行

      try
      {
         level2();//跳到level2方法处，执行level2()
      }
      catch (ArithmeticException problem)//捕捉异常，继续执行下面的语句
      {
         System.out.println ();
         System.out.println ("The exception message is: " +
                             problem.getMessage());
         System.out.println ();

         System.out.println ("The call stack trace:");
         problem.printStackTrace();
         System.out.println ();
      }

      System.out.println(“Level 1 ending.”); // Executed !执行结束后返回上一级方法main函数
   }

   public void level2()
   {
      System.out.println("Level 2 beginning.");//执行
      level3 ();//跳到level3方法处，调用level3()
      System.out.println(“Level 2 ending.”);//未捕捉异常，此处Not executed !返回调用该方法的上一级方法level1
   }
public void level3 ()
   {
      int numerator = 10, denominator = 0;//执行

      System.out.println("Level 3 beginning.");//执行
      int result = numerator / denominator;//0作除数，抛出异常
      System.out.println("Level 3 ending.");// 未捕捉异常，此处Not executed !，返回调用该方法level3的上一级方法level2
   }
}
```
>结果：

``` javascript
Program beginning.
Level 1 beginning.
Level 2 beginning.
Level 3 beginning.

The exception message is: / by zero

The call stack trace:
java.lang.ArithmeticException: / by zero
	at ExceptionScope.level3(ExceptionScope.java:54)
	at ExceptionScope.level2(ExceptionScope.java:41)
	at ExceptionScope.level1(ExceptionScope.java:18)
	at Propagation.main(Propagation.java:17)

Level 1 ending.
Program ending.

```