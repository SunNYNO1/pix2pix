import tensorflow as tf
from config import parse_args

args = parse_args()

def resize(img):
    r_img = tf.image.resize_images(img, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, 1)), dtype=tf.int32)
    if args.scale_size > args.crop_size-1:
        r_img = tf.image.crop_to_bounding_box(r_img, offset[0], offset[1], args.crop_size, args.crop_size)
    elif args.scale_size < args.crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r_img