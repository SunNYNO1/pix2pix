# pix2pix学习

### 整体规划（一个月）

> 1、数据迭代器
>
> 2、构建网络框架
>
> 3、创建优化器



## 2019/4/9

### 一周备忘录

> 构建pix2pix的数据迭代器

- 1、了解pix2pix的输入（阅读源代码）
- 2、迭代器的输出是什么
- 3、自己写一遍数据迭代器（参考连接：https://www.jianshu.com/p/eec32f6c5503）
- 4、测试自己的数据迭代器



> 今日任务

- [x] 了解pix2pix的输入及输出（8点30前）
- [x] 大概看一遍参考链接
- [ ] 写一遍自己的数据迭代器

> **了解pix2pix的输入及输出**

* ```python
  decode = tf.image.decode_jpeg  #将jpeg图片解码为unit8的张量
  os.path.basename()方法  #返回path最后的文件名
  ```

  

  * 例：

    >```python
    >os.path.basename('c:\test.csv')
    >>> 'test.csv'
    >```

* ***Line279：move image from [0,1] to [-1,1] ???***

* pix2pix输入数据基本操作
  * 读取路径，保存至列表

  * 对列表内的路径排序

  * 将图片分割为两张图片A,B

  * 将图片大小resize并crop为256*256

  * ```python
    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)   #批读取路径、输入图像、目标图像
    ```

    

  * 输入图像输入到生成器；输入图像与输出图像，或输入图像与目标图像输入到判别器

* tf.train.ExponentialMovingAverage(）#指数滑动平均：用于更新变量，decay是衰减率，一般设置为接近1的值，比如0.99、0.999，值越大更新越慢

> **Tensorflow的数据读取**

* **tf.data.TFRecordDataset && make_one_shot_iterator()**

  * ```python 
    tf.FixedLenFeature()  #用于读取数据前的样本解析，样本解析需要其构建解析字典，以分析每条样本的各个特征属性，除了tf.FixedLenFeature(）之外，还有VarLenFeature，FixedLenSequenceFeature等，依次表示定长特征、可变长特征、定长序特征
    ```

    

  * ```python
    parsed = tf.parse_single_example(参数1：单个样本，参数2：解析字典） #用于解析单个样本
    ```

    

  * ```python
    tf.cast()  #用于解析单个样本
    ```

    

  * ```python
    tf.data.TFRecordDataset(.tfrecords文件的路径)  #读取tfrecords的内容
    ```

    

  * dataset操作的一些函数：
    * ```python
      dataset = dataset.map(某一函数） #对dataset中的所有样本执行该函数,返回值为数据集
      dataset = dataset.shuffle(2)#将数据打乱，数值越大，混乱程度越大
      dataset = dataset.batch(4)#按照顺序取出4行数据，最后一次输出可能小于batch
      dataset = dataset.repeat()#数据集重复了指定次数（repeat在batch操作输出完毕后再执行）
      ```

      

  * ```python
    dataset.make_one_shot_iterator()  #迭代的读取数据集中每个样本，只迭代一次
    ```

    

* **tf.data.TFRecordDataset() & Initializable iterator**

  * 可初始化迭代器，显示的初始化（就是要自己去初始化），正因为如此，所以可以改变迭代器的数据集；相反，make_one_shot_iterator() 会自动初始化，也就没办法通过重新初始化来更换数据集？
  * sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})  #显式初始化
  * 可以用该迭代器更换验证集和训练集

* **tf.data.TextLineDataset() && Reinitializable iterator**

  * tf.decode.csv(line,num)  #相当于tf.parse_single_example(，解析单个样本的每个特征属性及标签

  * tf.data.TextLineDataset(txt或者csv文件路径)

  * 可以通过多个不同的 Dataset 对象进行初始化（但是，这些不同数据集的数据应该具有相同的shape和type）

  * 读取数据后，流程：

    * ```
      iterator = tf.data.Iterator.from_structure(training_dataset.output_types,                                            training_dataset.output_shapes)
      ```

      

    * ```
      features, labels = iterator.get_next()
      ```

      

    * ```python
      training_init_op = iterator.make_initializer(training_dataset)
      validation_init_op = iterator.make_initializer(validation_dataset) # 初始化两个不同的数据集
      ```

      

    * ```python
      sess.run(training_init_op)
      
      sess.run(validation_init_op) 
      ```

      

* **tf.data.TextLineDataset() & Feedable iterator**
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

## 2019/4/10

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

* 读取指定格式的文件路径，输出到路径列表

  ```python
  img_paths = glob.glob(filename+'*.jpg') 
  ```

* 已知图像数据的文件路径，使用以下方法读取图片

  ```python
  #方法一
  image_string = tf.read_file(filename)
  
  image_decoded = tf.image.decode_image(image_string)
  
  img =  tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  ```

  ```python
  #方法二
  将图片转换为tfrecord数据，使用昨天学习的tf.data.TFRecordDataset()读取数据，生成dataset
  ```

* pix2pix数据集的图片是图片对，需要分割为两张图片

  ```python
  img_a = image_decoded[:,:width//2,:]
  img_b = image_decoded[:,width//2:,:]
  ```

* 使用以下函数制作dataset

  ```python
  dataset = tf.data.Dataset.from_tensor_slices((imgs_a, imgs_b)) 
  #该函数用于切分传入图像的第一个维度，生成相应的dataset，也可以切分列表、字典，比如a=[1,2,3],b=[i,ii,iii],则该函数输出为[[1,i],[2,ii],[3,iii]]
  ```

  - 例子：  传入的是（5，2）的一个矩阵， tf.data.Dataset.from_tensor_slice会把它按第一个维度切分为5个元素，每个元素形状为（2，）

  ```python
  dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2))) 
  ```

* 创建数据迭代器

  ```python
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

    * ```python
      wget http://mirrors.aliyun.com/linux-kernel/people/tytso/e2fsprogs/v1.43.3/e2fsprogs-1.43.3.tar.gz
      tar -zxvf e2fsprogs-1.43.3.tar.gz
      cd e2fsprogs-1.43.3
      ./configure
      make  #如果有报错就sudo make，下同理
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



## 2019-4-11

> 今日计划

- [x] 英语
  - 单词
  - 阅读
  - 听力
- [x] ubuntu安装teamviewer
- [x] pix2pix数据迭代器：

- 先看以下博客：https://zhuanlan.zhihu.com/p/33223782
- 能够将图片转换为tfrecord文件，并保存图片文件，思路清晰
- 仿照朱代码框架，修改其代码写一个读取pix2pix的数据迭代器
- 测试自己的代码

- [ ] 深度学习视频
- [x] 组会
- [ ] <font color=red>英语试题一套</font>

> pix2pix数据迭代器-构建思路

* **数据预处理（针对pix2pix，需要将图片一分为二）**

```python
#读取原图路径
import glob
def read_img_paths(filename):
    img_paths = glob.glob(filename+'*.jpg')
    if len(img_paths) == 0:
        img_paths = glob.glob(filename+'*.png')
    print (img_paths)
    return img_paths
```

```python
#读入一张图片，一分为二
def single_img_split_ny(img_path):
	#根据路径读取图片
    img = cv.imread(img_path)
    #一分为二
    width = int(img.shape[1])
    img_a = img[:,:width//2,:]
    img_b = img[:,width//2:,:]
    return img_a,img_b
```

> > 读取图片的方式
> >
> > * 使用tf
> >   * **tf.gfile.FastGFile()**
> >   * **tf.train.string_input_producer()+tf.WholeFileReader().read()**
> >   * **tf.read_file()**
> >   * **TFRecords**
> > * **opencv: cv2.imread**
> > * **PIL：PIL.Image.open**
> > * **matplotlib：matplotlib.image.imread**
> > * **scipy.misc：scipy.misc.imread**
> > * **skimage：skimage.io.imread**

```python
#将所有图片一分为二，并分别保存在两个列表
def img_split(img_paths):
    imgs_a = []
    imgs_b = []
    for img_path in img_paths:
        img_a,img_b = single_img_split_ny(img_path)
        imgs_a.append(img_a)
        imgs_b.append(img_b)
    return imgs_a,imgs_b
```

* **将图像数据转换为tfrecords文件**

```python
def create_tfrecord(img_paths):
    imgs_a,imgs_b = img_split(img_paths)
    imgs_a = np.array(imgs_a)
    imgs_b = np.array(imgs_b)
    #利用字典将数据写入tfrecords文件
    features = {}
    #通过tf.python_io.TFRecordWriter将数据写入tfrecord文件中
    writer = tf.python_io.TFRecordWriter(filename+"%s.tfrecords" %"test")
    
    #将每张图片保存到tfrecords中，但tensorflow feature类型只接受list数据，这里可以通过以下方式：
    #转成list类型：将张量fatten成list(也就是向量)，再用写入list的方式写入。
	#转成string类型：将张量用.tostring()转换成string类型
    #因为丢失了形状信息，所以也要将形状保存在字典中，一起存入tfrecords，以便使用时恢复图像（张量）形状
    
    #使用tf.train.Feature函数将数据存入features字典中
    #存储类型如下：
    #int64：tf.train.Feature(int64_list = tf.train.Int64List(value=输入))
    #float32：tf.train.Feature(float_list = tf.train.FloatList(value=输入))
    #string：tf.train.Feature(bytes_list=tf.train.BytesList(value=输入))
    for i in range(len(imgs_a)):
        features['input_img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs_a[i].tostring()]))
        features['input_img_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=imgs_a[i].shape))                                                        
        features['target_img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs_b[i].tostring()]))
        features['target_img_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=imgs_b[i].shape))
        
    #     转成tfrecords.features
        tf_features = tf.train.Features(feature=features)
    #     生成example
        tf_examples = tf.train.Example(features = tf_features)    
    #     序列化样本
        tf_serialized = tf_examples.SerializeToString()
    #     写入样本
        writer.write(tf_serialized)
#     关闭tfrecords文件
    writer.close()
```

> >tfrecords创建流程：
> >
> >* 读取数据
> >* 创建字典，保存数据的各特征
> >* 通过writer=tf.python_io.TFRecordWriter将数据写入tfrecord文件中
> >* 将指定类型数据存入features字典中：tf.train.Feature(int64_list = tf.train.Int64List(value=列表类型的输入))
> >* 将字典转为tensorflow feature：tf.train.Features(feature=features)
> >* 将tensorflow feature生成example：tf.train.Example(features = tf_features)
> >* 将example序列化：tf_examples.SerializeToString()
> >* 将序列化example写入tfrecords：writer.write(tf_serialized)
>
> 
>
> >遇到的问题：
> >
> >* 在上面for循环时，部分代码写成下面这样，导致只读入了最后一条数据（自己理解错误导致）
> >
> >```python
> >for i in range(len(imgs_a)):
> >        features['input_img'] =  tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs_a[i].tostring()]))
> >        features['input_img_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=imgs_a[i].shape))                                                        
> >        features['target_img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs_b[i].tostring()]))
> >        features['target_img_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=imgs_b[i].shape))  
> >
> >#就是下面的代码
> >tf_features = tf.train.Features(feature=features)
> >tf_examples = tf.train.Example(features = tf_features)    
> >tf_serialized = tf_examples.SerializeToString()
> >writer.write(tf_serialized)
> >##就是上面的代码
> >```

* 执行一下上面的函数

```
filename = 'C:\\project\\PY\\GAN\\pix2pix-tensorflow-master\\pix2pix-tensorflow-master\\photos\\test\\'
# 分割图片
img_paths = read_img_paths(filename)
create_tfrecord(img_paths)
```

* **创建解析函数，解析tfrecords文件**

```python
def parse_function(example_proto):
    dic = {
        'input_img' : tf.FixedLenFeature(shape=(),dtype=tf.string,default_value=None),
        'input_img_shape' : tf.FixedLenFeature(shape(3),dtype=tf.int64,default_value=None),
        'target_img' : tf.FixedLenFeature(shape=(),dtype=tf.string,default_value=None),
        'target_img_shape' :  tf.FixedLenFeature(shape(3),dtype=tf.int64,default_value=None)
    }
    
    parse = tf.parse_single_example(example_proto,dic)
    
    parse['input_img']= tf.decode_raw(parse['input_img'],tf.uint8)
    parse['input_img'] = tf.reshape(parse['input_img'],parse['input_img_shape'])
    parse['target_img'] = tf.decode_raw(parse['target_img'],tf.uint8)
    parse['target_img'] = tf.reshape(parse['target_img'],parse['input_img_shape'])
    return parse
#     return {"input_img":input_img},{"output_img":output_img}
```

> > 创建解析函数流程：
> >
> > * 根据tfrecords内的数据结构与特征，创建对应的解析字典<font color=red>（这里就用到了昨天讲的tf.FixedLenFeature，即feature的解析方式）</font>，如果写入的feature使用了.tostring() 其shape就是()，否则shape属性就写对应特征的shape
> >
> > * 将上面建好的解析字典与一条tfrecords数据输入到tf.parse_single_example(example_proto,dic)，进行解析该条数据
> >
> > * 转变特征（通过parse['字典的key']可以调用数据，如果使用了下面两种情况，则还需要对这些值进行转变。）
> >
> >   * string类型：tf.decode_raw(parsed_feature, type) 来解码
> >
> >     * 这里type必须要和当初.tostring()化前的一致。如tensor转变前是tf.uint8，这里就需是tf.uint8；转变前是tf.float32，则tf.float32
> >
> >   * VarLenFeature解析：由于得到的是SparseTensor，所以视情况需要用tf.sparse_tensor_to_dense(SparseTensor)来转变成DenseTensor
> >
> >      parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.uint8)
> >
> >      稀疏表示 转为 密集表示
> >
> >      parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
> >
> > * 恢复形状
> >
> > ```
> > parse['input_img'] = tf.reshape(parse['input_img'],parse['input_img_shape'])
> > ```
> >
> > * 返回你想要的数据

* **使用dataset读取tfrecords**

  * 读入数据，创建dataset<font color=red>(用到昨天看的TFRecordDataset)</font>

  ```python
  dataset = tf.data.TFRecordDataset(刚生成的tfrecords数据的路径) 
  #或者直接读取非tfrecord数据
  #dataset = tf.data.Dataset.from_tensor_slices([1,2,3])
  ```

  * 对tfrecords中的全部数据进行解析

  ```python
  dataset = dataset.map(parse_function)
  ```

  * 创建迭代器<font color=red>(用到了昨天的make_initializable_iterator等)</font>

  ```python
  iterator = dataset.make_initializable_iterator()
  ```

  *  获取样本

    ```python
    dic = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        i = 1
        while True:
            # 不断的获得下一个样本
            try:
                # 获得的值直接属于graph的一部分，所以不再需要用feed_dict来喂+
                input_img,target_img = sess.run([dic['input_img'],dic['target_img']])
            # 如果遍历完了数据集，则返回错误
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
            else:
                # 显示每个样本中的所有feature的信息，只显示scalar的值
                print('==============example %s ==============' %i)
                print('input_img: value: %s | shape: %s' %(input_img, input_img.shape))
                print('input_img: value: %s | target_img shape: %s ' %(target_img, target_img.shape))
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(input_img)
                plt.subplot(1,2,2)
                plt.imshow(target_img)
            i+=1
    ```

    > > 同时绘制多个图像的方法:
    > >
    > > ```python
    > > plt.figure()
    > > plt.subplot(1,2,1)
    > > plt.imshow(input_img)
    > > plt.subplot(1,2,2)
    > > plt.imshow(target_img)
    > > ```

>今日总结

* 未完成：深度学习视频、英语试题、pix2pix世雄代码参考、代码检测
* 完成：单词、ubuntu安装teamviewer、组会、pix2pix流程、

> 明日计划

- [ ] 9：30前：英语
- [ ] pix2pix
  - [ ] 代码代码检测
  - [ ] 世雄代码对比
- [ ] 12点前：网络架构搭建
  - [ ] 待安排
- [ ] 2：30前：深度学习视频
- [ ] 5点前：新论文（deblurGAN、PGGAN）
- [ ] 6：30前：英语题
- [ ] 网络架构搭建

## 2019-4-12

> 今日计划

- [ ] ~~9：30前：英语~~
- [x] pix2pix
  - [x] 世雄代码对比
  - [x] 写成类，代码检测
  - [x] 写parse.config
  - [x] 写到py文件中
- [ ] 网络架构搭建
  - [ ] 理清pix2pix模型结构（generator、discriminator）
  - [ ] 自己编写模型
- [ ] ~~2：30前：深度学习视频~~
- [ ] 5点前：新论文（deblurGAN、~~PGGAN~~）
- [ ] 6：30前：英语题

> PIX2PIX

* 使用config

  ```
  import argparse
  def parse_args():
  
      parser = argparse.ArgumentParser(description='Tensorflow implementation of pix2pix')
  
      parser.add_argument('--module', type=str, default='test_dataset',
                          help='Module to select: train, test ...')
      parser.add_argument('--GPU', type=str, default='0',
                          help='The number of GPU used')
  
      return parser.parse_args()
  ```

  * 在parse_args()函数中调用parser = argparse.ArgumentParser()
  * 通过parse.add_argument('--name',type,default,help=’描述‘)，增加参数
  * 通过args=parse.parse_args()调用增加的参数，比如args.module，比如在main.py中使用

  ```python
  #main.py
  
  from config import parse_args
  from dataloader import Dataset
  
  args = parse_args()
  
  if __name__ == '__main__':
      #***使用args部分***
      module = args.module
      #***使用args部分***
      if module == 'create_tfrecords':
          img_paths = dataset.read_img_paths()
      else:
          print("This module has not been created!")
  ```

  * 在调用main.py时，输入如下命令，即可传入指定参数

  ```python
  python main.py--module create_tfrecords
  ```

  

* 代码模块化，将模块化的函数集成到类里面

  * 将已写好的函数直接移入类

    * 包含以下几个类

      ```python
      def read_img_paths(self)
      def read_img(self,img_paths)
      def single_img_split_ny(self,img)
      def img_split(self,img_paths)
      def create_tfrecord(self,img_paths)
      def parse_function(self,one_example_tfrecords)
      def load_dataset(self,tfrecords_list)
      def test_dataset(self)
      ```

      * 划分类的时候保证

      > <font color = red>每个函数不要太长</font>
      >
      > <font color = red>逻辑清晰</font>
      >
      > <font color = red>尽量使代码泛化能力强,能够复用</font>

  * 更改create_tfrecords函数，将数据写到多个tfrecords文件中

    * 为什么要将数据写到多个tfrecords中呢

      > 因为如果将数据都读入到一个文件中，在读取tfrecords文件时会一次性调用过多的内存，当数据特别大的时候，不能保证内存够用

    * 实现细节

      > * 定义一个全局列表，用来存储每个tfrecords文件的路径
      >
      >   ```
      >   self.TFRecords_List = []
      >   ```
      >
      >   
      >
      > * 定义两个参数，分别用来指定：总共生成多少个tf文件、每个tf文件共多少条数据（这两个参数可以根据自己数据量计算得出）
      >
      >   ```
      >   self.NUM_TFRECORDS = 100
      >   self.NUM_FILES_IN_TFRECORDS = 7
      >   ```
      >
      > * 通过两个循环，将指定数据依次读入到各个tf文件中
      >
      >   ```python
      >           num_tfrecords = 0
      >           i = 0
      >           while num_tfrecords<self.NUM_TFRECORDS :
      >               tf_filename = self.FILENAME+"tfrecords\\%03d.tfrecords" %num_tfrecords
      >               self.TFRecords_List.append(tf_filename)
      >               with tf.python_io.TFRecordWriter(tf_filename) as writer:
      >                   num_files_in_tfrecords = 0
      >                   while num_files_in_tfrecords<self.NUM_FILES_IN_TFRECORDS and i<len(imgs_a):
      >                       sys.stdout.write('\r>> num_files_now_tfrecords: %d/%d num_tfrecords:%d/%d' % (num_files_in_tfrecords+1, self.NUM_FILES_IN_TFRECORDS,num_tfrecords+1,self.NUM_TFRECORDS))
      >                       sys.stdout.flush()
      >                       '''
      >                       将数据各个特征读入解析字典的细节部分
      >                       '''
      >                       num_files_in_tfrecords+=1
      >                       i+=1
      >               num_tfrecords+=1
      >           writer.close()
      >   ```

    * 遇到一个特别傻的问题：在测试代码时报错，tfrecords都已经生成，但是最后因为tf.data.TFRecordDataset()写成了tf.TFRecordReader()报错(长得有点像，还是对常用的API不够熟悉)

> 网络架构搭建

* **理清网络架构**

  * **generator**

    ![1555295911845](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555295911845.png)

  * **discriminator**

    ![1555296007222](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555296007222.png)

* **其他问题**

  * 卷积边长计算：
    * 通用计算公式计算公式：new_width = (width-kernel_size+2*padding)/stride+1
      * 当padding='same'，padding补边公式：（kernel_size-1）/2，然后向下取整
      * 当padding='valid',不补边，多余的边舍弃掉
    * 专用公式：
      * padding=same专用公式：new_width = width/2,然后向上取整
      * padding=valid专用公式：new_width=(W–K+1) /stride,向上取整
    * <font color=red>卷积核大小为3*3，步长为1时，长度不变</font>

  * tf.concat(tensor1，tensor2，axis=1)，沿着某一维拼接张量

    ```
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    ```

  * tf.pad():

    ```
    
    t = tf.constant([[1, 2], [3, 4]])
    paddings = tf.constant([[2, 1], [1, 1]])
     
    a = tf.pad(t, paddings, "CONSTANT")
     
    sess = tf.Session()
    print(sess.run(a))
     
    结果如下:
    [[0 0 0 0]
     [0 0 0 0]
     [0 1 2 0]
     [0 3 4 0]
     [0 0 0 0]]
    --------------------- 
    作者：GodWriter 
    来源：CSDN 
    原文：https://blog.csdn.net/GodWriter/article/details/85244486 
    ```

    > padding中每个[a, b]都代表在相应的维度前后加上指定行数的0，比如例子中：[2, 1]指的是第0维（即行所在维度）的前面加2行0，后面加一行0；[1, 1]指的是在第1维（即列所在维度）前面加上1行0，后面加上1行0



## 2019-4-15

### 一周备忘录（2019-4-15）

- [x] 1、写pix2pix网络结构
- [ ] 2、完成pix2pix模型优化器部分
  - 理清solver模块思路，及模块间关系
  - test代码
  - 完善solver模块
- [x] ~~3、训练完整的pix2pix模型，实现1920*1080输出~~

- [ ] 其他

  * 一篇论文-deblur（论文主要思想+~~源码阅读~~）

  * 信息论作业

  * 周报

> 今日计划

- [x] 周报
- [x] 搭建网络
- [ ] 深度学习视频

> 搭建网络

* 主要代码(构建生成器、判别器)

  ```python
  def create_generator(input_img,output_channels):
      
      #***encoder***    
      conv1 = tf.layers.conv2d(input_img, filter=64, kernel_size = 4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      lrelu1 = tf.nn.leaky_relu(conv1,alpha=0.2)
      
      conv2 = tf.layers.conv2d(lrelu1, filter=64*2, kernel_size = 4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm2 = tf.layers.batch_normalization(conv2, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
      lrelu2 = tf.nn.leaky_relu(norm2, alpha=0.2)
      
      conv3 = tf.layers.conv2d(lrelu2, filter=64*4, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm3 = tf.layers.batch_normalization(conv3,axis=3,epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
      lrelu3 = tf.nn.leaky_relu(norm3,alpha=0.2)
      
      conv4 = tf.layers.conv2d(lrelu3, filter=64*8, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm4 = tf.layers.batch_normalization(conv4, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu4 = tf.nn.leaky_relu(norm4, alpha=0.2)
      
      conv5 = tf.layers.conv2d(lrelu4, filter=64*8, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm5 = tf.layers.batch_normalization(conv5, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu5 = tf.nn.leaky_relu(norm5, alpha=0.2)
      
      conv6 = tf.layers.conv2d(lrelu5, filter=64*8, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm6 = tf.layers.batch_normalization(conv6, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu6 = tf.nn.leaky_relu(norm6, alpha=0.2)
      
      conv7 = tf.layers.conv2d(lrelu6, filter=64*8, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm7 = tf.layers.batch_normalization(conv7, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu7 = tf.nn.leaky_relu(norm7, alpha=0.2)
      
      conv8 = tf.layers.conv2d(lrelu7, filter=64*8, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
      norm8 = tf.layers.batch_normalization(conv8, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      
      
      #***decoder***
      derelu1 = tf.nn.relu(norm8)
      deconv1 = tf.layers.conv2d_transpose(derelu1, 64*8, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm1 = tf.layers.batch_normalization(deconv1, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      dedrop1 = tf.nn.dropout(denorm1, keep_prob=1 - 0.5)
      
      
      deconcat2 = tf.concat(dedrop1,norm7,axis=3)
      derelu2 = tf.nn.relu(deconcat2)
      deconv2 = tf.layers.conv2d_transpose(derelu2, 64*8, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm2 = tf.layers.batch_normalization(deconv2,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      dedrop2 = tf.nn.dropout(denorm2,keep_prob=0.5)
      
      deconcat3 = tf.concat(dedrop2,norm6,axis=3)
      derelu3 = tf.nn.relu(deconcat3)
      deconv3 = tf.layers.conv2d_transpose(derelu3,64*8, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm3 = tf.layers.batch_normalization(deconv3,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      dedrop3 = tf.nn.dropout(denorm3,keep_prob=0.5)
      
      deconcat4 = tf.concat(dedrop3,norm5,axis=3)
      derelu4 = tf.nn.relu(deconcat4)
      deconv4 = tf.layers.conv2d_transpose(derelu4,64*8, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm4 = tf.layers.batch_normalization(deconv4,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      
      deconcat5 = tf.concat(dedrop4,norm4,axis=3)
      derelu5 = tf.nn.relu(deconcat5)
      deconv5 = tf.layers.conv2d_transpose(derelu5,64*4, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm5 = tf.layers.batch_normalization(deconv5,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      
      deconcat6 = tf.concat(dedrop5,norm3,axis=3)
      derelu6 = tf.nn.relu(deconcat6)
      deconv6 = tf.layers.conv2d_transpose(derelu6,64*2, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm6 = tf.layers.batch_normalization(deconv6,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      
      deconcat7 = tf.concat(dedrop6,norm2,axis=3)
      derelu7 = tf.nn.relu(deconcat7)
      deconv7 = tf.layers.conv2d_transpose(derelu7,64, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm7 = tf.layers.batch_normalization(deconv7,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      
      deconcat8 = tf.concat(dedrop7,norm1,axis=3)
      derelu8 = tf.nn.relu(deconcat8)
      deconv8 = tf.layers.conv2d_transpose(derelu8,3,kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
      denorm8 = tf.layers.batch_normalization(deconv8,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      #源代码自定义的tanh函数
      detanh8 = tf.nn.tanh(denorm8)
      
      return detanh8 
  ```

  ```python
  def create_discriminator(input_img,target_img):
      
      input_target_img = tf.concat(input_img,target_img,axis=3)
      
      conv_pad1 = tf.pad(input_target_img,[[0,0],[1,1],[1,1],[0,0]],mode = "CONSTANT")
      conv1 = tf.layers.conv2d(conv_pad1, 64, kernel_size = 4, strides=(2,2), padding='valid', kernel_initialize = tf.random_normal_initializer(0,0.02))
      lrelu1 = tf.nn.leaky_relu(conv1,0.2)
      
      conv_pad2 = tf.pad(lrelu1,[[0,0],[1,1],[1,1],[0,0]],axis=3)
      conv2 = tf.layers.conv2d(copnv_pad2, 64*2, kernel_size=4, strides=(2,2),padding='valid', kernel_initializer=tf.random_normal_initializer(0,0.02))
      norm2 = tf.layers.batch_normalization(conv2,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu2 = tf.nn.leaky_relu(norm2,0.2)
      
      conv_pad3 = tf.pad(lrelu2,[[0,0],[1,1],[1,1],[0,0]],axis=3)
      conv3 = tf.layers.conv2d(copnv_pad3, 64*4, kernel_size=4, strides=(2,2),padding='valid', kernel_initializer=tf.random_normal_initializer(0,0.02))
      norm3 = tf.layers.batch_normalization(conv3,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu3 = tf.nn.leaky_relu(norm3,0.2)
      
      conv_pad4 = tf.pad(lrelu3,[[0,0],[1,1],[1,1],[0,0]],axis=3)
      conv4 = tf.layers.conv2d(copnv_pad4, 64*8, kernel_size=4, strides=(1,1),padding='valid', kernel_initializer=tf.random_normal_initializer(0,0.02))
      norm4 = tf.layers.batch_normalization(conv4,axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0,0.02))
      lrelu4 = tf.nn.leaky_relu(norm4,0.2)
      
      conv_pad5 = tf.pad(lrelu4,[[0,0],[1,1],[1,1],[0,0]],axis=3)
      conv5 = tf.layers.conv2d(copnv_pad5, 1, kernel_size=4, strides=(1,1),padding='valid', kernel_initializer=tf.random_normal_initializer(0,0.02))
      sigmoid5 = tf.nn.sigmoid(conv5)
      
      return sigmoid5
  ```

  

* 如何实现卷积/反卷积：（注意：这里使用的是tf.layers函数，不是tf.nn函数）

  ```python
  tf.layers.conv2d(input_img, out_channels, 
                   kernel_size = 4, strides=(2,2), 
                   padding='SAME'， 
                   kernel_initializer=tf.random_normal_initializer(1.0, 0.02))
  ```

  ```python
  tf.layers.conv2d_transpose(batch_input, out_channels, 
                             kernel_size=4, strides=(2, 2), 
                             padding="same", 
                             kernel_initializer=tf.random_normal_initializer(0, 0.02))
  ```

  

* 如何实现批归一化：

  ```python
  tf.layers.batch_normalization(layer_output,axis=3, 
                                epsilon=1e-5, momentum=0.1, 
                                training=True, 
                                gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
  ```

* 常用激活函数：

  * relu/sigmoid/tanh

  ```python
  tf.nn.relu(inputdata)
  tf.nn.sigmoid(input_data)
  tf.nn.tanh(input_data)
  ```

  * leaky_relu

  ```python
  tf.nn.leaky_relu(input_data,alpha)
  ```

* x = tf.identity（)函数作用**（实质：复制张量并建立节点）**

  * 通过在计算图内部创建 send （发送）/ recv（接收）节点来**引用或复制变量**；

  * 最主要的用途：**在不同设备间传递变量的值**

  * **与tf.control_dependencies(）配套使用**，tf.control_dependencies的意义如下代码注释

  * 举个例子：

    ```python
    x = tf.Variable(1.0)
    y = tf.Variable(0.0)
    
    #返回一个op，表示给变量x加1的操作
    x_plus_1 = tf.assign_add(x, 1)
    
    #control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前，
    #先执行control_dependencies参数中的内容（在这里就是 x_plus_1）
    with tf.control_dependencies([x_plus_1]):
        y = x
    init = tf.initialize_all_variables()
    
    with tf.Session() as session:
        init.run()
        for i in xrange(5):
            print(y.eval())#相当于sess.run(y)
    ```

    ```python
    x = tf.Variable(1.0)
    y = tf.Variable(0.0)
    x_plus_1 = tf.assign_add(x, 1)
    
    with tf.control_dependencies([x_plus_1]):
        #***使用了tf.identity***
        y = tf.identity(x)
        #***使用了tf.identity***
        
    init = tf.initialize_all_variables()
    
    with tf.Session() as session:
        init.run()
        for i in xrange(5):
            print(y.eval())
    --------------------- 
    原文：https://blog.csdn.net/hu_guan_jie/article/details/78495297 
    ```

    > * 对于control_dependencies这个管理器，只有当里面的操作是一个op时，才会生效；其先执行
    > * 传入的参数op，再执行里面的op
    >
    > * y=x与tf.identity区别：
    >   * y=x仅是tensor的一个简单赋值，不是定义的op，所以在图中不会形成一个节点，这样该管理器就失效了。
    >   * tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中，这时control_dependencies就会生效，所以输出符合预期：2，3，4，5，6

> 今日总结

* 将pix2pix模型图示
* 编写pix2pix网络架构（参考网络架构图，可以在源码的参考下写出其网络架构，但是对常用的API参数设置不清楚；其次，思路思路混乱，没有清楚的将结构记在脑子里，所以时不时的参考架构图和源码，导致思路混乱）
* 今天时间利用率有点低，其实可以完成更多的工作（比如优化器部分、deblur论文），自我要求和自我约束不够

> 明日计划

- [ ] 9：30前：英语
- [ ] 2点30前：深度学习视频
- [ ] pix2pix优化器
- [ ] pix2pix模块化
- [ ] deblur论文

---



## 2019-4-16

> 今日计划

- [x] pix2pix优化器
- [ ] 2点30前：深度学习视频
- [ ] pix2pix模块化
- [ ] deblur论文

> pix2pix优化器

* **构建损失函数**

  * 判别器损失：

    ```python
    tf.reduce_mean(-(tf.log( D(input_img，target_img) + epsilon ) + tf.log(1-D(inputimg，gengeraor_img) + epsilon )))
    ```

  * 生成器损失：

    ```python 
    #生成器第一部分损失
    gen_loss = tf.reduce_mean(-(tf.log(D(generator_img)+epsilon))
    #L1损失
    l1_loss = tf.reduce_mean(tf.abs(target_img - generator_img))
                   
    #总的生成器损失
    gen_l1_loss = gen_wight*gen_loss + l1_wight*l1_loss               
    ```

  * 代码与原文有些区别：

    - 原文损失没有Epsilon参数

    - 原文中的期望求解E()：tf.reduce_mean()

    - 原文中的损失公式：

      > 总损失：

      ![1555385529488](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555385529488.png)

      

      > 生成损失和判别损失放在一起：

      ![1555385379993](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555385379993.png)

      > L1损失：

      

      ![1555385428112](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555385428112.png)

      

* **列出需要更新的变量列表**

  * 在create_generator、create_discriminator函数中的变量前，**使用with tf.veriable_scope()函数，为变量指定前缀**

  * 使用var.name.startswith("discriminator"）指定变量进行更新

    ```python
    dis_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")
    ```

* **优化器更新参数**

  * 初始化optimizer

    ```python
    optimizer = tf.train.AdamOptimizer( learning_rate, beta1,beta2)
    
    #***其他优化器***
    
    #GradientDescentOptimizer 
    #AdagradOptimizer 
    #AdagradDAOptimizer 
    #MomentumOptimizer 
    #FtrlOptimizer 
    #RMSPropOptimizer
    
    #***其他优化器***
    --------------------- 
    #原文：https://blog.csdn.net/xierhacker/article/details/53174558 
    ```

    >  这里的beta1表示：第一个动量的衰减率

  * 使用optimizer更新参数

    ```python
    train = optimizer.minimize(loss, varlist)
    #或者
    grads_and_vars = gen_optim.compute_gradients(loss, var_list)
    gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
    ```

    > **optimizer.minimiz（）等价于**
    >
    > **grads_and_vars = optimizer.compute_gradients(gen_loss, var_list=gen_tvars)**
    >
    > **train = optimizer.apply_gradients(gen_grads_and_vars)的加和**

* **使用指数滑动平均（EMA）（是否就是指数加权平均）:**

  ```python
  ema = tf.train.ExponentialMovingAverage(decay=0.99)
  update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])#为这些变量设置影子变量
  ema.average([变量名列表])#获取变量的影子变量
  ```

  > * 所谓影子变量，就是对原变量的复制，使得原变量更新的时候，影子变量依然维持原值
  > * 用于更新参数、变量
  >
  > * 可以和其他优化算法结合使用
  > * [参考链接1](https://www.jianshu.com/p/2f53606d4b6d)、[参考链接2](http://www.cnblogs.com/hellcat/p/8583379.html)

* **其他**：

  * 设置global_step:

    ```
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    ```

  * tensorflow中的赋值：

    ```python
    tf.assign（A,B）#将B的值赋值给A
    ```

> 今日总结

* 在写优化器的时候，没有理清逻辑，思路比较混乱，也没有和之前写的几个模块联系在一起，没有写完优化器；所以接下来的主要任务是对比世雄的代码和源代码，<font color=red>写一个存储图片的模块用于保存生成图像；理清各个模块之间的联系；写test代码</font>，**先完成基本的训练功能，在考虑使用tensorboard、保存model**
* 执行计划上，一直拖在第一条，其他任务也没有完成

## 2019-4-17

> 今日计划

- [x] 理清solver模块思路，及模块间关系
- [x] 实验室服务
- [ ] deblur
- [x] 和世雄讨论代码问题，接下来的安排
- [ ] 1深度学习视频

> 理清solver模块思路，及模块间关系

* 报错corrupted record at 0

  * 由于路径设置错误，没有成功导入数据所致

* 反卷积的原理，及边长计算公式

  * ouput = (input-1)*s+k-2p

* 使用尺寸不一的图片：

  * 修改placeholder的shape，设置长宽为None

  * concat报错：

    * 报错的shape[0]是指什么

    - 已经知道input_img、target_img的shape=[512\*384*3]，所以shape[0]shape[1]指的不是两张图片

    ```python
    input_target_img = tf.concat([input_img,target_img],axis=3)#concat报错
    
    #报错内容
    ConcatOp : Dimensions of inputs should match: shape[0] = [1,3,512,512] vs. shape[1] = [1,3,512,384]
    ```

    * 对concat函数进行测试

    ```python
    a = [[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]]
    b = [[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]]
    
    print(np.array(a).shape)
    
    c = tf.concat([a,b],axis=1)
    with tf.Session() as sess:
        m = sess.run(c)
        print(m.shape)
    ```

    * 输出：(2, 2, 2, 3)，(2, 4, 2, 3)

    * 结论：axis=1是对通道进行合并，axis=0，1，2，3，依次对应<font color=red>batch，通道，长，宽</font>；并不是batch，宽，长，通道

    * 疑问：而原文是axis=3，那么是对宽度进行合并？？？

      > 原文中依然是对通道的合并，至于为什么上面测试与原文的axis对应关系不同，可能是因为data_format的原因

      > 关于data_format
      >
      > > 在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧，也即**Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32）**，**Caffe**采取的也是这种方式。
      > >
      > > **而TensorFlow**，的表达形式是**（100,16,32,3）**
      > >
      > > ```python
      > > #keras
      > > Convolution2D(
      > >     batch_input_shape=(None, 1, 28, 28),   #多少数据  通道数  宽  高
      > >     filters=32,     #滤波器数量
      > >     kernel_size=5,   #滤波器大小5x5
      > >     strides=1,       #步长1
      > >     padding='same',     # Padding method
      > >     data_format='channels_first',
      > > )
      > > #在卷积函数中指定data_format为'channels_first'，即通道优先即可
      > > ```
      > >
      > > ```python
      > > #tensorflow数据格式转换
      > > '''
      > > NHWC 
      > > [batch, in_height, in_width, in_channels]
      > > NCHW 
      > > [batch, in_channels, in_height, in_width]
      > > '''
      > > 
      > > #NHWC –> NCHW
      > > x = tf.reshape(tf.range(24), [1, 3, 4, 2])
      > > out = tf.transpose(x, [0, 3, 1, 2])
      > > print x.shape
      > > print out.shape
      > > (1, 3, 4, 2)
      > > (1, 2, 3, 4)
      > > 
      > > #NCHW –> NHWC
      > > x = tf.reshape(tf.range(24), [1, 2, 3, 4])
      > > out = tf.transpose(x, [0, 2, 3, 1])
      > > print x.shape
      > > print out.shape
      > > (1, 2, 3, 4)
      > > (1, 3, 4, 2)
      > > ```
      > >
      > > 

    - 修改axis=1，依然报错，猜想报错的原因可能和图片尺寸不一

> 今日总结

* 完成了最简单的train函数，对256*256的数据可以跑通，但是当输入不同尺寸就会报错，刚开始以为是axis设置参数不正确，但修改之后依然报同样的错误



## 2019-4-18

> 今日计划

- [ ] pix2pix
  - [x] 更换统一尺寸的数据集进行训练看是否报错
  - [x] 更换统一尺寸的X*X矩形图片进行训练
  - [ ] 如果实验依旧不成功，先在256*256的基础上，把test部分写好，完善代码（test、save_model、<font color=red>summary(tensorboard的使用)</font>）
  - [ ] ~~完善代码后，写resize函数，对输入图片（1920\*1080）缩放到合适尺寸（2048*2048）~~
  - [ ] ~~[更改网络模型](https://github.com/karolmajek/pix2pix-tensorflow/blob/master/pix2pix.py#L262)，将缩放后的图片输入到模型中训练~~
  - [ ] ~~再将输出的图片，重新缩放到原尺寸（1920*1080）~~
- [ ] 中午：deblur

> 更换统一尺寸的数据集进行训练看是否报错（384*512）

* 依然在以下地方报错：

  ```python
  #generator.py
  deconcat2 = tf.concat([dedrop1,norm7],axis=3)
  ```

  * 通过打印dedrop1，norm7的形状，发现这两个形状不一致

  ```python
  print('dedrop1', dedrop1.shape,'norm7:',norm7)
  ```

  * 于是检查生成模型网络结构，发现384*512的尺寸输入到8层encoder中（每层输出都会除以2），导致
    * 第7层之后（shape为[3，4]）没有办法整除，到了第8层其将会缩放到[4,4]
    * 进入decoder之后进行反卷积，最后的输出会变成[512,512]
    * 而这里的[512*512]的输出则是生成器的输出(即generator_img)，当计算L1损失的时候会计算target_img-generator_img，但是前者是[384\*512]，而后者是[512\*512]，所以报错

* debug的关键：

  * 从报错处入手，打印出来查看结果

> deblur

* 论文内容：

  * 数据集：GOPRO

  * 主要贡献：

    * 提出模糊化的LOSS函数与模型结构
    * 随机轨道生成法生成模糊数据集
    * 用于去模糊的新的评估算法（基于提高目标检测率）

  * 生成模糊图像的原理：

    * 使用运动轨迹随机生成方法<font color=red>(马尔可夫随机过程)</font>
    * 对轨迹"子像素插值"，以生成blur kernel

  * LOSS:

    * L = Lgan + Ladv

    * Lgan用的就是WGAN-GP的损失函数

    * Ladv称之为Propatual loss（感知损失)

      > 基于目标图像与生成图像特征图的差

  * 网络结构（类似pix2pix）

    * <font color=red>WGAN-GP体现在哪里？（是体现在Lgan Loss上吗）</font>
    * <font color=red>Resout结构是什么样子的？与unet类似？</font>

  ![1555636795180](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555636795180.png)

  ![1555636822122](C:\Users\12466\AppData\Roaming\Typora\typora-user-images\1555636822122.png)

  

> 今日总结

* 弄清楚了pix2pix网络结构及其特殊性，一些细微的思想还要看原论文
* 基于高清图像进行训练的想法不可行，因为没有高清数据集作为训练集，只改变测试图像的输入输出大小没有意义
  * 基于以上问题可以有两种解决方案
    * 1、找到1024*1024或者2048\*2048的高清训练集，然后再按照上面的想法接着做
    * 2、将当前的训练集图片resize成512*512的图像进行训练，结合sniper对其512\*512的切片作为输入，从而实现高分辨率的图像数据增强<font color=red>(小论文思路)</font>



## 2019-4-19

> 今日计划

- [x] 写resize函数，将训练集resize到512*512
- [ ] 中午：pix2pix论文（博客、原论文）
- [ ] 下午：deblur代码

## 2019-4-22

### 一周备忘录

* idea

  > - [ ] 世雄的风格迁移模块代码
  > - [ ] 将风格迁移加入pix2pix中

* 完善solver模块（周三前）

  > - [ ] 加入summary
  > - [ ] 保存model
  > - [ ] 保存图片
  > - [ ] test

* pix2pix论文、cGAN论文，ppt

* PGGAN论文

* 复制粘贴的项目

* 周末：

  > - [ ] 信息论作业
  > - [ ] 周报

> 今日计划

- [ ] pix2pix博客，论文
- [ ] 完善solver模块

> pix2pix论文

* bottleneck layer
* patchGAN
* U-NET