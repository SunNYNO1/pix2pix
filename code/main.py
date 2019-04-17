import os
from config import parse_args
from dataloader import Dataset
from solver import Solver

args = parse_args()

dataset = Dataset(args)
solver = Solver(args)

if __name__ == '__main__':
    module = args.module
    if module == 'create_tfrecords':
        img_paths = dataset.read_img_paths()
        dataset.create_tfrecord(img_paths)
    elif module == 'test_dataset':
        dataset.test_dataset()
    elif module == 'train':
        solver.train()
    else:
        print("This module has not been created!")