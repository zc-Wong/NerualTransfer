# **人工智能大作业-----图像风格迁移**

## 一、项目简介

​		所谓图像风格迁移，是指利用算法学习一种图片的风格，然后再把这种风格应用到另外一张图片上的技术。具体来讲如下图所示：

![img](F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/20eac1a6f86640b085a78732f7e6ca7f/1573747780%281%29.png)

​								图片来自论文Image Style Transfer Using Convolutional Neural Networks

​		本次课程大作业参考上述论文使用VGGNet以及TensorFlow框架实现了图像风格转移，并使用python与javascript在短时间内搭建了一个测试网站。

## **二、开发工具**

### **2.1 算法实现工具**

​		**算法实现平台是Anaconda与Jupyternotebook**。**Anaconda**指的是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。 **Jupyter Notebook** 是一款开放源代码的 Web 应用程序，可让我们创建并共享代码和文档。它提供了一个环境，你可以在其中记录代码，运行代码，查看结果，可视化数据并在查看输出结果。这些特性使其成为一款执行端到端数据科学工作流程的便捷工具 ，可以用于数据清理，统计建模，构建和训练机器学习模型，可视化数据以及许多其他用途。

​		**算法实现框架基于TensorFlow**。TensorFlo是一个基于数据流编程的符号数学系统，被广泛应用于各类机器学习算法的编程实现，其前身是谷歌的神经网络算法库DistBelief 。Tensorflow拥有多层级结构，可部署于各类服务器、PC终端和网页并支持GPU和TPU高性能数值计算，被广泛应用于谷歌内部的产品开发和各领域的科学研究

### **2.2 测试网站开发工具**

​		**网站的开发使用Pycharm。**PyCharm是一种Python IDE，带有一整套可以帮助用户在使用Python语言开发时提高其效率的工具，比如调试、语法高亮、Project管理、代码跳转、智能提示、自动完成、单元测试、版本控制。此外，该IDE提供了一些高级功能，以用于支持Django框架下的专业Web开发。

​		**框架使用Flask。**Flask是一个使用 Python 编写的轻量级 Web 应用框架。其 WSGI 工具箱采用 Werkzeug ，模板引擎则使用 Jinja2 。Flask使用 BSD 授权。Flask也被称为 “microframework” ，因为它使用简单的核心，用 extension 增加其他功能。Flask没有默认使用的数据库、窗体验证工具。

## **三、开发环境**

### **3.1 算法开发环境**

操作系统：Windows 10

开发环境：Jupyter Notebook

运行环境：Anaconda环境下Python3.6

### **3.2 网页开发环境**

操作系统：Windows 7

服务器：本机

开发环境：Pycharm

## **四、平台分析**

### **4.1 进入主页**

​		在Pycharm中，点击运行start.py文件，flask框架搭建一个小的后台，点击控制台中的

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/deb8d149ff1444a881dc046be681b421/1573805300%281%29.png)

​		即可进入首页。

### **4.2 首页界面**

​		进入首页后，可以看到如下图所示界面：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/a06cb843739142478b3bf07f247b4cfb/1573805579%281%29.png)

​		主页有20种风格的图片，开始必须选择一种风格，如果选择直接点击下一步，则会提醒必须选择一张图片，如下图所示：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/6d7a93ca21b94cf9bb5e0f68ee491dfa/1573805703%281%29.png)

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/075498ff799549f2a99f39949b482499/1573805981%281%29.png)

​		只能选择一种风格，如果多选了，则会提醒，如下图所示：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/1d356f7d546d41549d03729ae68a655a/1573807417%281%29.png)

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/047418e949f343a39c38eb05850abeea/1573807451%281%29.png)

​		正确操作之后进入下一个界面，如下图所示：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/bf36910f32b44255a63f07a8a956c180/1573807531%281%29.png)

​		左边的界面功能未实现，右边界面点击选择文件，可以选定内容图片，与前面的风格图片进行风格融合：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/33531e5e432b4e2c93286085173ad69f/1573807887%281%29.png)

### **4.3 图像风格融合**

​		点击开始，后台就开始进行风格融合，如下图所示：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/1f7df70c5bca4d2382c7aa890e5099f2/1573807967%281%29.png)

​		最后得到融合后的新图像：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/cb1854b85c4e4304896b50de48c2eff5/cdb689fe60d1f6379b9c5d3107968e8.png)

​		在指定的文件夹下，可以找到融合后的新图片：

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/16534e13231b47b09d712df6fb4d7d59/outputpp_at_iteration_8.png)

## **五、程序设计**

### **5.1 算法的实现**

![img](F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/1472d83e38284ef7a6b7251c8682f582/1573783041%281%29.png)

​																					算法原理示意图

​		1890年，梵高看到他将画架画星空的那个晚上附近时发生了什么？如果发明了彩色照片，则可能记录了那个美妙的夜晚。莫奈通过淡淡的画笔描边和调色板表达了他对同样场景的印象。如果莫奈在一个凉爽的夏日夜晚发生在西湖上，该怎么办？短暂浏览梵高画作的画廊，可以想象他将如何渲染场景：也许是柔和的阴影，突然的油漆刷和稍微平淡的动态范围。我们可以想象所有这一切，尽管从未在他所画的场景照片旁边看到莫奈画作的并排示例。取而代之的是，我们了解了莫奈的绘画和风景照。我们可以推断出这两个集合之间的风格差异，从而可以想象如果将场景从一个集合“转换”到另一个集合中，场景将是什么样子。

#### **5.1.1 特征工程**

```python
width, height = load_img(base_image_path).size
# 行
img_nrows = 200
# 列
img_ncols = int(width * img_nrows / height)

# 图片预处理
def preprocess_image(image_path):
    # 使用Keras内置函数读入图片并设置为指定长宽
    img = load_img(image_path, target_size=(img_nrows, img_ncols)) 
    # 转为numpy array格式
    img = img_to_array(img) 
    #：keras中tensor是4维张量，所以给数据加上一个维度
    img = np.expand_dims(img, axis=0) 
    # vgg提供的预处理，主要完成（1）减去颜色均值
    #（2）RGB转BGR
    #（3）维度调换三个任务。
    # 减去颜色均值可以提升效果
    # RGB转BGR是因为这个权重是在caffe上训练的，caffe的彩色维度顺序是BGR。
    # 维度调换是要根据系统设置的维度顺序tensorflow将通道维调到正确的位置
    img = vgg16.preprocess_input(img) 
    return img
```

​		在深度学习中，大家都会发现训练集，验证集合测试集划分好之后会有减去均值的一步操作，但很多人都是只跟着做，并没有探究为什么要做这一步处理。

​		其主要原理是我们默认自然图像是一类平稳的数据分布(即数据每一维的统计都服从相同分布)，此时，在每个样本上减去数据的统计平均值可以移除共同的部分，凸显个体差异。其效果如下所示：                   

![img](file:///F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/3905427912d644c5a978fa3de4289028/1573791435%281%29.png)

​																			图像减去色彩均值示意图

​		可以看到天空的纹理被移除了，凸显了汽车和高楼等主要特征。

#### **5.1.2 模型选择**

```python
# 载入模型 
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False) 
```

算法的实现采用VGG16模型对图像风格和内容进行特征提取。

![img](F:/%E7%A0%94%E7%A9%B6%E7%94%9F%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/qq713072305C062177C841124677D61BB7/9bb9c6b55e294331804d5db086de119b/clipboard.png)

​																				VGGNet结构示意图

​		VGG是由Simonyan 和Zisserman在文献《Very Deep Convolutional Networks for Large Scale Image Recognition》中提出卷积神经网络模型，其名称来源于作者所在的牛津大学视觉几何组(Visual Geometry Group)的缩写。

​		该模型参加2014年的 ImageNet图像分类与定位挑战赛，取得了优异成绩：在分类任务上排名第二，在定位任务上排名第一。

​		VGGNet突出贡献在于证明使用很小的卷积（3*3），增加网络深度可以有效提升模型的效果，而且VGGNet对其他数据集具有很好的泛化能力。到目前为止，VGGNet依然经常被用来提取图像特征。

#### **5.1.3 风格Loss计算**

```python
# 设置Gram矩阵的计算图，首先用batch_flatten将输出的feature map扁平化，
# 然后自己跟自己的转置矩阵做乘法，跟我们之前说过的过程一样。
#注意这里的输入是深度学习网络某一层的输出值。
def gram_matrix(x):
    # permute_dimensions按照给定的模式重排一个张量
    # batch_flatten将一个n阶张量转变为2阶张量，其第一维度保留不变，
    # 这里的扁平化主要是保留特征图的个数，让二维的特征图变成一维(类似上图)
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # 格拉姆矩阵
    gram = K.dot(features, K.transpose(features))
    return gram

# 设置风格loss计算方式，以风格图片和待优化的图片的某一卷积层的输入作为输入。
# 计算他们的Gram矩阵，然后计算两个Gram矩阵的差的平方，除以一个归一化值
def style_loss(style, combination): 
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2)) 
```



#### **5.1.4 内容Loss计算**

```python
# 设置内容loss计算方式，以内容图片和待优化的图片的representation为输入，计算他们差的平方。像素级对比
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 施加全变差正则，全变差正则化常用于图片去噪，可以使生成的图片更加平滑自然。
def total_variation_loss(x): 
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
    b = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
```



#### **5.1.5 梯度下降**

```python
# 获取loss和grads
def eval_loss_and_grads(x):
    # 把输入reshape层矩阵
    x = x.reshape((1, img_nrows, img_ncols, 3))
    # 这里调用了我们刚定义的计算图
    outs = f_outputs([x])
    loss_value = outs[0]
    # outs是一个长为2的tuple，0号位置是loss，1号位置是grads。
    #把grads扁平化
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


# 定义了两个方法，一个用于返回loss，一个用于返回grads
class Evaluator(object):
    def __init__(self):
        # 初始化损失值和梯度值
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        # 调用函数得到梯度值和损失值，但只返回损失值，
        #而将梯度值保存在成员变量self.grads_values中
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        # 这个函数不用做任何计算，只需要把成员变量
        #self.grads_values的值返回去就行了
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
```



## **六、课程小结**

### **6.1 遇到的问题**

- 算法不能做到实时出成果，是需要改进的地方
- 只有风格相近的或者内容相近的图像，融合后效果才好，例如风格图像的内容是山水的，那么内容图像是山水，融合效果才好，这也是以后需要改进的地方

### **6.2 未来需要改进的地方**

- 把应用移植到移动端，更好的适应市场
- 进行算法优化，力求做到实时出成果
- 尽可能的实现多张图像风格融合技术

## 七、参考资料

2016Cvpr:Image Style Transfer Using Convolutional Neural Networks