## Pix2pix学习

### 整体规划（一个月）

> 1、数据迭代器
>
> 2、构建网络框架
>
> 3、创建优化器

### 一周备忘录

> 构建pix2pix的数据迭代器

* 1、了解pix2pix的输入（阅读源代码）
* 2、迭代器的输出是什么
* 3、自己写一遍数据迭代器（参考连接：https://www.jianshu.com/p/eec32f6c5503）
* 4、测试自己的数据迭代器

### 2019/4/9

> 今日任务

- [x] 了解pix2pix的输入及输出（8点30前）
- [x] 大概看一遍参考链接
- [ ] 写一遍自己的数据迭代器

> **了解pix2pix的输入及输出**

* decode = tf.image.decode_jpeg  #将jpeg图片解码为unit8的张量

* os.path.basename()方法  #返回path最后的文件名

  * 例：

    >》》os.path.basename('c:\test.csv')
    >'test.csv'

* ***Line279：move image from [0,1] to [-1,1] ???***

* pix2pix输入数据基本操作
  * 读取路径，保存至列表
  * 对列表内的路径排序
  * 将图片分割为两张图片A,B
  * 将图片大小resize并crop为256*256
  * paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)   #批读取路径、输入图像、目标图像
  * 输入图像输入到生成器；输入图像与输出图像，或输入图像与目标图像输入到判别器

* tf.train.ExponentialMovingAverage(）#指数滑动平均：用于更新变量，decay是衰减率，一般设置为接近1的值，比如0.99、0.999，值越大更新越慢

> **Tensorflow的数据读取**

* **tf.data.TFRecordDataset && make_one_shot_iterator()**
  * tf.FixedLenFeature()  #用于读取数据前的样本解析，样本解析需要构建解析字典，以分析每条样本的各个特征属性，除了tf.FixedLenFeature(）之外，还有VarLenFeature，FixedLenSequenceFeature等，依次表示定长特征、可变长特征、定长序特征
  * ​    parsed = tf.parse_single_example(参数1：单个样本，参数2：解析字典） #用于解析单个样本
  * tf.cast()  #用于解析单个样本
  * tf.data.TFRecordDataset(.tfrecords文件的路径)  #读取tfrecords的内容
  * dataset操作的一些函数：
    * datset = dataset.map(某一函数） #对dataset中的所有样本执行该函数,返回值为数据集
    * dataset = dataset.shuffle(2)#将数据打乱，数值越大，混乱程度越大
    * dataset = dataset.batch(4)#按照顺序取出4行数据，最后一次输出可能小于batch
    * dataset = dataset.repeat()#数据集重复了指定次数（repeat在batch操作输出完毕后再执行）
  * dataset.make_one_shot_iterator()  #迭代的读取数据集中每个样本，只迭代一次

* **tf.data.TFRecordDataset() & Initializable iterator**

  * 可初始化迭代器，显示的初始化（就是要自己去初始化），正因为如此，所以可以改变迭代器的数据集；相反，make_one_shot_iterator() 会自动初始化，也就没办法通过重新初始化来更换数据集？
  * sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})  #显式初始化
  * 可以用该迭代器更换验证集和训练集

* **tf.data.TextLineDataset() && Reinitializable iterator****

  * tf.decode.csv(line,num)  #相当于tf.parse_single_example(，解析单个样本的每个特征属性及标签

  * tf.data.TextLineDataset(txt或者csv文件路径)

  * 可以通过多个不同的 Dataset 对象进行初始化（但是，这些不同数据集的数据应该具有相同的shape和type）

  * 读取数据后，流程：

    * iterator = tf.data.Iterator.from_structure(training_dataset.output_types,                                            training_dataset.output_shapes)

    * features, labels = iterator.get_next()

    * training_init_op = iterator.make_initializer(training_dataset)
      validation_init_op = iterator.make_initializer(validation_dataset) # 初始化两个不同的数据集

    * sess.run(training_init_op)

      sess.run(validation_init_op) 

  * #### tf.data.TextLineDataset() & Feedable iterator.

  * 流程：

    * 如上过程读取文件、创建好数据集
    * 创建一个placeholder，用来传输handle（句柄）
    * 使用iterator = tf.data.Iterator.from_string_handle（handle，training_dataset.output_types, training_dataset.output_shapes），创建一个可以转换handle（句柄）的迭代器
    * 从迭代器读取特征和标签，features, labels = iterator.get_next()
    * 使用make_one_shot_iterator或者make_initializable_iterator对每个数据集创建一个对应的迭代器：training_iterator = training_dataset.make_one_shot_iterator()
    * 将上一步的迭代器转为handle（句柄），training_handle = sess.run(training_iterator.string_handle())
    * 将这些句柄依次传入创建好的placeholder，sess.run(labels, feed_dict={handle: training_handle})

> 今日总结

* 前两项任务完成，但第二项任务看的不是很懂
* 自己的数据迭代器编写没有完成，第二项任务（阅读数据迭代器参考资料）上花掉太多时间

> 明日计划

* 写一遍自己的数据迭代器
* 测试自己的迭代器
* 了解pix2pix网络架构

### 2019/4/10

> 今日任务
>
> > 上午

- [ ] 9点30前：英语单词、阅读、听力
- [x] 12点前：写一遍自己的数据迭代器

> > 下午

- [x] 2点前：深度学习视频

- [x] 5点前：复制粘贴

- [ ] 6点30前：深度学习视频

- [x] 8点30前：复制粘贴、讨论数据迭代器的问题

- [ ] 9点30前：英语试题




> 编写自己的数据迭代器

* 

*  读取指定格式的文件路径，输出到路径列表

  ```
  img_paths = glob.glob(filename+'*.jpg') 
  ```

* 已知图像数据的文件路径，使用以下方法读取图片

  ```python
  #方法一
  image_string = tf.read_file(filename)
  
  image_decoded = tf.image.decode_image(image_string)
  
  img =  tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  ```

  ```
  
  ```

  

  ```python
  #方法x
  将图片转换为tfrecord数据，使用昨天学习的tf.data.TFRecordDataset()读取数据，生成dataset
  ```

* pix2pix数据集的图片是图片对，需要分割为两张图片

  ```python
  img_a = image_decoded[:,:width//2,:]
  img_b = image_decoded[:,width//2:,:]
  ```

* 使用以下函数制作dataset

  ```
  dataset = tf.data.Dataset.from_tensor_slices((imgs_a, imgs_b)) 
  #该函数用于切分传入图像的第一个维度，生成相应的dataset，也可以切分列表、字典，比如a=[1,2,3],b=[i,ii,iii],则该函数输出为[[1,i],[2,ii],[3,iii]]
  ```

  - 例子：  传入的是（5，2）的一个矩阵， tf.data.Dataset.from_tensor_slice会把它按第一个维度切分为5个元素，每个元素形状为（2，）

  ```
  dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2))) 
  ```

* 创建数据迭代器

  ```
  iterator = dataset.make_one_shot_iterator()
  或
  iterator = dataset.make_initializable_iterator()
  ```

* 从迭代器中读取数据

  ```
  i,j= iterator.get_next()
  ```

  * 显示图片	

> 复制粘贴环境配置（ubuntu系统下）

* ubuntu根分区不够----根分区扩容

  * [参考链接1](https://blog.csdn.net/qq_38265674/article/details/80523820)

    [参考链接2](https://zhuanlan.zhihu.com/p/32937122)

  * 在u盘启动的ubuntu中进行根分区扩容时，报错：Get a newer  version of *e2fsck*。

    * `打开终端`-输入`e2fsck -V`查看 *e2fsck*版本(注意`V`需要大写）

    * 若版本<1.43.3，执行如下安装步骤

    * ```
      wget http://mirrors.aliyun.com/linux-kernel/people/tytso/e2fsprogs/v1.43.3/e2fsprogs-1.43.3.tar.gz
      tar -zxvf e2fsprogs-1.43.3.tar.gz
      cd e2fsprogs-1.43.3
      ./configure
      make  #如果有报错就sudo make， 下同理
      make install
      ```

  * 

> 今日总结

* 走到太晚，再加上制定当天计划，单词没背
* 数据迭代器
  * 只是东拼西凑的实现了功能，没有自己的套路，逻辑不清晰
  * 没有将所学的内容活学活用，应该将二者联系起来
  * 应该理清楚思路，将代码写得更有条理，形成自己的套路，才能越写越熟练
* 深度学习视频
  * 没有将前后知识贯通
  * 将重点做下笔记
  * 系统的认识知识点
* 复制粘贴模型
  * 主要收获：从下午5点解决根目录分区的问题，用掉了整晚的时间
  * 缺陷：
    * ubuntu环境配置流程不熟悉（<font color=red>nvidia驱动安装、cuda安装、</font>cudnn安装）
    * 复制粘贴数据预处理、模型的执行流程依然不清楚

>明日计划

- [ ] 英语
  - 单词
  - 阅读
  - 听力
- [ ] ubuntu安装teamviewer

- [ ] pix2pix数据迭代器：

* 先看以下博客：https://zhuanlan.zhihu.com/p/33223782
* 能够将图片转换为tfrecord文件，并保存图片文件
* 仿照朱代码框架，修改其代码写一个读取pix2pix的数据迭代器
* 测试自己的代码

- [ ] 深度学习视频

- [ ] 组会

- [ ] <font color=red>英语试题一套</font>



## 2019-4-10

> 今日计划

