import tensorflow as tf
from config import parse_args as args

class Discriminator():
    def __init__(self,args):
        self.args = args

    def create_discriminator(self,input_img,target_img):
        
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