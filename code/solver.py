from config import parse_args  as args
from dataloader import Dataset
from generator import create_generator 
from discriminator import create_discriminator 
import tensorflow as tf


def train():
    
    epsilon = 1e-12
    #生成器
    output_channels = int(target_img.get_shape()[-1])
    generator = create_generator(input_img,output_channels)

    input_img = tf.placeholder(shape=[args.batch_size,img_wight,img_hight,3],dtype=tf.float32)
    target_img = tf.placeholder(shape=[args.batch_size,img_wight,img_hight,3],dtype=tf.float32)
    #关于真实图片的判别器
    real_discriminator = create_discriminator(input_img,target_img)
    
    #关于生成图片的判别器
    fake_discriminator = create_discriminator(generator,target_img)
    
    #生成器损失
    gen_gan_loss =  tf.reduce_mean(-tf.log(fake_discriminator+epsilon))
    gen_l1_loss = tf.reduce(tf.abs(target_img-generator))
    gen_loss = args.gen_wight*gen_gan_loss + args.l1_wight*gen_l1_loss
    #判别器损失
    dis_loss = tf.reduce_mean(-(tf.log(real_discriminator+epsilon)+tf.log(1-fake_discriminator+epsilon)))
    
    #生成器需要更新的变量列表
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    #判别器需要更新的变量列表
    dis_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    
    #生成器优化器
    gen_optimizer = tf.train.AdamOptimizer(0.0002,0.5)
    #判别器优化器
    dis_optimizer = tf.train.AdamOptimizer(0.0002,0.5)
    
    #使用损失 对指定变量进行训练更新
    #与使用optimizer.minimize()区别？？？？
    gen_gradients = gen_optimizer.compute_gradients(gen_loss,var_list=gen_vars)
    gen_train = gen_optimizer.apply_gradients(gen_gradients)
    dis_gradients = dis_optimizer.compute_gradients(dis_loss,var_list=dis_vars)
    dis_train = dis_optimizer.apply_gradients(dis_gradients)
    
    #更新参数
    # ema = tf.train.ExponentialMovingAverage(decay=0.99)
    # update_losses = ema.apply([dis_loss, gen_gan_loss, gen_l1_loss])
    
    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step,global_step+1)

    dataset = Dataset()
    one_img = dataset.load()

    init_global = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_global)
        while True:
                #不断的获得下一个样本
                try:
                    input_img,target_img = sess.run([one_img[input_img],one_img[target_img],gen_train,dis_train,gen_loss,dis_loss],feed_dict={input_img:input_img,target_img:target_img})
                #如果遍历完了数据集，则返回错误
                except tf.errors.OutOfRangeError:
                    print("End of dataset")
                    break
       
def test():
