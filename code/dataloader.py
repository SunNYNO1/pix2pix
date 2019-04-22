import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import sys
from config import parse_args as args
import os

class Dataset():
    
    
    def __init__(self,args):
        self.TFRecords_List = []

        #***需要根据数据集修改***
        self.NUM_TFRECORDS = 1
        self.NUM_FILES_IN_TFRECORDS = 5
        #***需要根据数据集修改***

        self.args = args
        
        #***tfrecords数据保存路径***
        self.FILENAME = 'C:\\project\\PY\\GAN\\pix2pix-tensorflow-master\\pix2pix-tensorflow-master\\rain\\facades\\train_test\\'
        #***tfrecords数据保存路径***
        
    
    #读取所有图像路径到列表
    def read_img_paths(self):
        img_paths = glob.glob(self.FILENAME+'*.jpg')
        if len(img_paths) == 0:
            img_paths = glob.glob(self.FILENAME+'*.png')
        return img_paths
    
    
    #使用cv2.imread读取图像
    def read_img(self,img_paths):
        imgs = []
        for img_path in img_paths:
            img = cv.imread(img_path)
            imgs.append(img)
        return imgs
    
    
    #***用于pix2pix模型分割图像***
    def single_img_split_ny(self,img):
        width = int(img.shape[1])
        img_a = img[:,:width//2,:]
        img_b = img[:,width//2:,:]
        return img_a,img_b
  

    #***用于pix2pix模型分割图像***
    def img_split(self,img_paths):
        imgs_a = []
        imgs_b = []
        imgs = self.read_img(img_paths)
        for img in imgs:
            img_a,img_b = self.single_img_split_ny(img)
            imgs_a.append(img_a)
            imgs_b.append(img_b)
        return imgs_a,imgs_b
    
    
    #将原始数据写入tfrecord
    def create_tfrecord(self,img_paths):
        
        #***下面部分适用于pix2pix模型***
        #如果是其他模型，以下部分替换为：通过图像路径列表，将图像保存到图像列表中
        imgs_a,imgs_b = self.img_split(img_paths)
        imgs_a = np.array(imgs_a)
        imgs_b = np.array(imgs_b)
        #***上面部分适用于pix2pix模型***
        
        features = {}
        num_tfrecords = 0
        i = 0
        while num_tfrecords<self.NUM_TFRECORDS :
            tf_filename = self.FILENAME+"tfrecords\\%03d.tfrecords" %num_tfrecords
            self.TFRecords_List.append(tf_filename)
            with tf.python_io.TFRecordWriter(tf_filename) as writer:
                num_files_in_tfrecords = 0
                while num_files_in_tfrecords<self.NUM_FILES_IN_TFRECORDS and i<len(imgs_a):
                    sys.stdout.write('\r>> num_files_now_tfrecords: %d/%d num_tfrecords:%d/%d' % (num_files_in_tfrecords+1, self.NUM_FILES_IN_TFRECORDS,num_tfrecords+1,self.NUM_TFRECORDS))
                    sys.stdout.flush()
                    
                    #***对应数据的特征结构，需要修改***
                    features['input_img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs_a[i].tostring()]))
                    features['input_img_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=imgs_a[i].shape))                                                        
                    features['target_img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs_b[i].tostring()]))
                    features['target_img_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=imgs_b[i].shape))
                    #***对应数据的特征结构，需要修改***
                    
                    #转成tfrecords.features
                    tf_features = tf.train.Features(feature=features)
                    #生成example
                    tf_examples = tf.train.Example(features = tf_features)    
                    #序列化样本
                    tf_serialized = tf_examples.SerializeToString()
                    #写入样本
                    writer.write(tf_serialized)
                    num_files_in_tfrecords+=1
                    i+=1
            num_tfrecords+=1
        writer.close()
        
        
    #解析tfrecord文件，并返回解析出的数据
    def parse_function(self,one_example_tfrecords):
        
        #***对应数据的特征结构，需要修改***
        dic = {
            'input_img' : tf.FixedLenFeature(shape=(),dtype=tf.string,default_value=None),
            'input_img_shape' :  tf.FixedLenFeature(shape=(3),dtype=tf.int64,default_value=None),
            'target_img' : tf.FixedLenFeature(shape=(),dtype=tf.string,default_value=None),
            'target_img_shape' :  tf.FixedLenFeature(shape=(3),dtype=tf.int64,default_value=None)
        }
        #***对应数据的特征结构，需要修改***
        
        parse = tf.parse_single_example(one_example_tfrecords,dic)
        
        #***对应数据的特征结构，需要修改***
        parse['input_img']= tf.decode_raw(parse['input_img'],tf.uint8)
        parse['input_img'] = tf.reshape(parse['input_img'],parse['input_img_shape'])
        parse['target_img'] = tf.decode_raw(parse['target_img'],tf.uint8)
        parse['target_img'] = tf.reshape(parse['target_img'],parse['input_img_shape'])
        #***对应数据的特征结构，需要修改***
        
        return parse
    
    
    #创建dataset及iterator，并返回iterator
    def load_dataset(self,tfrecords_list):
        imgpaths = self.read_img_paths()
        self.create_tfrecord(imgpaths)
        dataset = tf.data.TFRecordDataset(tfrecords_list)
        dataset = dataset.map(self.parse_function)
        iterator = dataset.make_initializable_iterator()
        return iterator
    
    
    #测试-读取iterator中的tfrecord数据并打印
    def test_dataset(self):
        tfrecords_list = tf.placeholder(tf.string,shape=None)
        iterator = self.load_dataset(tfrecords_list)
        dic = iterator.get_next()
        with tf.Session() as sess:
            sess.run(iterator.initializer,feed_dict={tfrecords_list:self.TFRecords_List})
            i = 1
            while True:
                #不断的获得下一个样本
                try:
                    #获得的值直接属于graph的一部分，所以不再需要用feed_dict来喂
                    
                    #***数据需要改的部分***
                    input_img,target_img = sess.run([dic['input_img'],dic['target_img']])
                    #***数据需要改的部分***
                    
                #如果遍历完了数据集，则返回错误
                except tf.errors.OutOfRangeError:
                    print("End of dataset")
                    break
                else:
                    #显示每个样本中的所有feature的信息，只显示scalar的值
                    print('==============example %s ==============' %i)
                    print('input_img: value: %s | shape: %s' %(input_img, input_img.shape))
                    print('input_img: value: %s | target_img shape: %s ' %(target_img, target_img.shape))
                    #plt.figure()
                    #plt.subplot(1,2,1)
                    #plt.imshow(input_img)
                    #plt.subplot(1,2,2)
                    #plt.imshow(target_img)
                i+=1
    
    def load(self):
        tfrecords_list = []

        for data in os.listdir(self.args.dataset):
            tfrecords_list.append(os.path.join(self.args.dataset, data))

        dataset = tf.data.TFRecordDataset(tfrecords_list)
        new_dataset = dataset.map(self.parse_function)
        shuffle_dataset = new_dataset.shuffle(buffer_size=len(tfrecords_list))
        batch_dataset = shuffle_dataset.batch(self.args.batch_size)
        epoch_dataset = batch_dataset.repeat(self.args.num_epochs)

        iterator = epoch_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return next_element