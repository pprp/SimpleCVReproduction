# MarkTex
MarkTex是将Markdown内容转换为Latex文档的Python库，如果熟悉一些LaTeX的话，那么该库应该是当前最好最美观排版最舒适可定制性最强的Markdown转PDF的方案。

## 特性汇总

1. 支持markdown中基本上所有的特性：标题、代码、引用、目录、图片、表格、链接...
1. 图片支持行内图片（会自动调整大小适应一行）和行间图片，支持使用本地相对路径和网络链接，会自动判断下载
2. 表格自动调整列宽，且进行了相关美化，不会变丑
3. 支持通过tex模板文件定制
4. 支持在当前markdown中引入其他markdown和tex文件，实现很方便的协作
...

 最新支持的全部语法可以在[example.md](./marktex/example/example.md)中参考，相应的效果可以查看[example.pdf](./output/out/example.pdf)，README中因为比较麻烦，更新可能不会很及时。


## 使用方式
```bash
% 因为pip源刚上传，所以用国内源可能会找不到
pip install marktex -i https://pypi.python.org/pypi
```

```python
from marktex.texrender import MarkTex

doc = MarkTex.convert_file("path/of/markdownfile","path/of/output_image/dir")
doc.generate_tex()
```

目录`outoput/`下的例子可以通过以下代码生成
```python
from marktex.example import run_example
run_example("./output/")
```


你也可以通过命令行运行：

输出到对应文件的 "文件名" 所在的目录下：
```bash
marktex a.md b.md ...
```

输出到一个同一的文件夹下：
```bash
marktex a.md b.md ... -o "path"
```

指定输出到各自文件夹，必须保证路径个数和文件个数相同：
```bash
marktex a.md b.md ... -e "pathfora" "pathforb" ...
```

## 特性介绍
具体可以参考[example.md](./marktex/example/example.md)
其pdf输出效果可以参考
[example.pdf](./output/example.pdf)

### 目录
```bash
 [toc]
```
![在这里插入图片描述](./src/toc.png)
### 特性介绍
```bash
# 特性<sub>下标在这里</sub>
- 支持目前主流的所有markdown语法（目前，脚注和xml标签暂时不支持）
- 额外添加了下划线语法（`__下划线__`）
- 表格自动调整列宽
- 复选框支持三种
- 无论是本地图片还是网络图片，都能够支持。
```
![在这里插入图片描述](./src/feature.png)
### 文字效果与五级标题
```bash
# 效果演示

本文用于演示和测试转换后的效果

## 普通文本
支持一般的文本和**加粗**，*斜体*，`行内代码`，和$InLine Formula$，[超链接](http://github.com)，注意公式暂时不支持中文。

~~删除线~~,__下划线__

## 二级标题

### 三级标题
目录编号支持到三级标题，可以通过修改latex文件或者直接更改模板来完成。

#### 四级标题
##### 五级标题
```

![在这里插入图片描述](./src/effect.png)

### 表格
可以完美的自适应表格列宽（测试效果良好，不排除特例），不过暂时不支持表格内插入图片
```bash
## 表格
支持一般的文本格式，暂时不支持表格内图片。另外，表格取消了浮动（float），因此不支持对表格的描述（caption），不过在Markdown中也没有对表格的描述，因此也不算功能不完善。

|ColA| ColB |
|--|--|
| **Table Bold** |  *Table Italic*|
| `Table Code` |  $Table Formula$|
|[Table line](www.github.com)|Table Text|

|A|B|C|Long Text Sample Long Text Sample Long Text Sample Long Text Sample Long Text Sample Long Text Sample |
|--|--|--|--|
|A|B|C|D|
|A|B|C|D|
|A|B|C|D|
```
![在这里插入图片描述](./src/table.png)

### 列表、序号、复选框
```bash
## 列表和序号/itemize&enumerate
- 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)
- 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)
- 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)

1. 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)
2. 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)
3. 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)

 [x] 支持
 [√] 三种
 [] 复选框格式
```
![在这里插入图片描述](./src/list.png)

### 图片
图片支持网络图片和本地图片，会被统一的哈希命名后存放到自定义的图片目录下
```bash
## 图片
和表格一样，取消了浮动，因此暂时不支持对图片的描述。不过本项目支持网络图片，会在转换的时候自动下载到本地。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190726170401866.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhaWxpc3Q=,size_16,color_FFFFFF,t_70)
```

![在这里插入图片描述](./src/img1.png)
![在这里插入图片描述](./src/img2.png)

### 公式
```bash

## 公式
公式不支持中文，并且没有编号
$$
f(x_i)=ax_i+b
$$
```
![在这里插入图片描述](./src/fomular.png)


### 代码

```bash
代码使用tcolorbox和minted，基本支持所有主流语言。支持的所有语言请参考 [Code Highlighting with minted](https://www.overleaf.com/learn/latex/Code_Highlighting_with_minted) 
```
```python
if __name__ == "__main__":
	print("hello world!")
```
```cpp
#include<stdio.h>
int main(){
	printf("hello world")
	return 0;
}
```

![](./src/code.png)

### 引用
```bash
## 引用
> 引用内环境和普通文本基本一致，但是不支持标题。
> 演示**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)
> - 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)
> 1. 支持**加粗**，*斜体*，`行内代码`,$Inline Formula$，[超链接](www.github.com)

> 表格：
> |ColA| ColB |
>|--|--|
>| **Table Bold** |  *Table Italic*|
>| `Table Code` |  $Table Formula$|
>|[Table line](www.github.com)|Table Text|
> 公式：
> $$F(x_i) = wx_i+b$$
> 图片：
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190726170401866.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhaWxpc3Q=,size_16,color_FFFFFF,t_70)
> 
```
![在这里插入图片描述](./src/quote.png)

![在这里插入图片描述](./src/quote2.png)

### 新特性
```bash
# 新特性-引入其他Markdown文档

非常酷的特性！可以使用特殊的html标签来引入其他的MarkDown！

<include>./table_example.md</include>

<include>./formula_example.md</include>

```

![](./src/newf.png)

![](./src/newf2.png)

## TODOs
 [x] 2019年7月29日:删除线和下划线的添加
 [x] 2019年7月29日:复选框的识别
 [x] 2019年7月29日:目录
 [x] 2019年7月30日:表格的美化
 [x] 2019年8月1日:支持xml标签的识别
 > 目前支持 `<title>`,`<author>`,`<sub>`,`<super>`，目前可以统一被分析到markdown的目录树，不过没有考虑好转换成tex的方式。
 
 [x] 2019年8月1日:图片相对路径的优化，更改了类的参数，图片将统一放到tex文件所在路径的`images`路径下，并在tex文件内统一使用相对路径进行表示
 [x] 封面
 [x] 2019年8月2日：通过引入<include>标签，支持多个markdown文件合并
 [] 水印
 [x] 2019年10月30日：正式支持四级和五级标题，已经通过paragraph和subparagraph来实现，但格式不是很好看，因此不推荐使用。
 [x] 代码环境美化(2019年10月25日完成)
 [x] 参数可定制化(可以通过修改tex模板来完成,2019年10月25日)
 [] 支持加粗、斜体、...这些语法的嵌套（放弃） 
 [] 添加对MarkDown直接支持但是LaTeX不支持的符号的转换如（θ）
 [x] 2019年10月30日:添加LaTeX混排，\<include\>标签内加入.md文件则引入md文件，加入.tex文件则引入tex文件，注意没有办法区分序言区，是完全复制粘贴的形式，使用方式请按照latex中include命令的使用方式，不要在文件中添加只在序言区生效的命令。
 [] 支持在线编译直接生成pdf（根据我的库[synctex](https://github.com/sailist/synctex)）
 
 ## 注意
有一些小的规范需要注意，否则转换可能会出错：
 - 引用环境会一直保持知道碰到第一行空行，因此单纯的不使用引用标记 > 是不好用的，需要空行
 - 目前不支持基本Token的嵌套，也就是说，加粗，斜体，代码这些是不能嵌套使用的，如果嵌套，会按代码中处理的优先级处理
 
