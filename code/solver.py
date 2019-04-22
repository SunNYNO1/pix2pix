from config import parse_args  
from dataloader import Dataset
from generator import create_generator 
from discriminator import create_discriminator 
import tensorflow as tf
from process import resize

args = parse_args()

def train():
    
    epsilon = 1e-12

    input_img = tf.placeholder(shape=[args.batch_size,512,512,3],dtype=tf.float32)
    target_img = tf.placeholder(shape=[args.batch_size,512,512,3],dtype=tf.float32)

    #生成器
    with tf.variable_scope("generator"):
        output_channels = int(target_img.get_shape()[-1])
        generator = create_generator(input_img,output_channels)

    #关于真实图片的判别器
    with tf.variable_scope('discriminator'):
        real_discriminator = create_discriminator(input_img,target_img)
    
    #关于生成图片的判别器
    with tf.variable_scope('discriminator',reuse = True):
        fake_discriminator = create_discriminator(input_img,generator)
    
    #定义生成器、判别器损失
    gen_gan_loss =  tf.reduce_mean(-tf.log(fake_discriminator+epsilon))
    gen_l1_loss = tf.reduce_mean(tf.abs(target_img-generator))
    gen_loss = args.gan_weight*gen_gan_loss + args.l1_weight*gen_l1_loss
    dis_loss = tf.reduce_mean(-(tf.log(real_discriminator+epsilon)+tf.log(1-fake_discriminator+epsilon)))
    
    #需要更新的变量列表
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    dis_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    #定义优化器
    global_step = tf.Variable(0,name='global_step',trainable=False)
    gen_optimizer = tf.train.AdamOptimizer(0.0002,0.5)
    dis_optimizer = tf.train.AdamOptimizer(0.0002,0.5)
    '''
    使用损失 对指定变量进行训练更新
    '''
    gen_gradients = gen_optimizer.compute_gradients(gen_loss,var_list=gen_vars)
    gen_train = gen_optimizer.apply_gradients(gen_gradients,global_step=global_step)
    dis_gradients = dis_optimizer.compute_gradients(dis_loss,var_list=dis_vars)
    dis_train = dis_optimizer.apply_gradients(dis_gradients,global_step=global_step)

    #存储模型
    # saver = tf.train.Saver(max_to_keep=1)

    #初始化所有变量
    init_global = tf.global_variables_initializer()

    '''
    #更新参数
    # ema = tf.train.ExponentialMovingAverage(decay=0.99)
    # update_losses = ema.apply([dis_loss, gen_gan_loss, gen_l1_loss])
    
    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step,global_step+1)
    '''
    dataset = Dataset(args)
    one_img = dataset.load()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_global)
        '''
        #加载已有的模型，断点训练
        #在config中添加self.args.retraining、self.args.model_path参数
        if self.args.retraining:
            last_model = tf.train.latest_checkpoint(self.args.model_path)
            print("Restoring model from {}".format(last_model))
            saver.restore(sess, last_model)
        '''
        '''
        #保存打印summary
        '''
        i = 1
        while True:
                #不断的获得下一个样本
                try:
                    input_img_,target_img_ = sess.run([one_img['input_img'],one_img['target_img']])
                    r_input_img = sess.run(resize(input_img_))
                    r_target_img = sess.run(resize(target_img_))
                    _, _, gen_loss_, dis_loss_ = sess.run([gen_train,dis_train,gen_loss,dis_loss],feed_dict={input_img:r_input_img,target_img:r_target_img})
                    print('gen_loss：',gen_loss_,'dis_loss:',dis_loss_)
                #如果遍历完了数据集，则返回错误
                except tf.errors.OutOfRangeError:
                    print("End of dataset")
                    break
                i += 1
def test():
    pass

train()