import tensorflow as tf
from config import parse_args as args


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