import argparse

def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Tensorflow implementation of pix2pix"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--module', type=str, default='test_dataset',
                        help='Module to select: train, test ...')
    parser.add_argument('--GPU', type=str, default='0',
                        help='The number of GPU used')
    parser.add_argument('--is_training', type=bool, default=True,
                        help='When the module is train, True, else False')
    parser.add_argument('--tfrecord_dir', type=str, default='data/VOC-tfrecord',
                        help='Path of the dataset transferred to tfrecord')
    parser.add_argument('--split', type=str, default='train',
                        help='Type of data to transfer: train, test ...')
    parser.add_argument("--l1_weight", type=float, default=100.0, 
                        help="weight on L1 term for generator gradient")#L1正则化项的权重参数
    parser.add_argument("--gan_weight", type=float, default=1.0, 
                        help="weight on GAN term for generator gradient")#生成器梯度的GAN的权重参数
    parser.add_argument("--dataset", type=float, default='C:\\project\\PY\\GAN\\pix2pix-tensorflow-master\\pix2pix-tensorflow-master\\rain\\facades\\train\\', 
                        help="Path of the dataset")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The size of each training batch')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='The number of the total training epoch')

    return parser.parse_args()