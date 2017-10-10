"""WIP Dataset compiler to be used on the cmd line, specifically in a unix environment.
#
Compiles jpg images in a given directory in an idx formatted file with the intention to be used in
the training or verification of a neural network. Files should all be the same dimensions as per a network would
input, and their naming convention should relate to their tag i.e x.jpg is name of a jpg containing x.
#
It is a fairly simple script with just three steps:
collect - collect all appropriate files in the given directory
convert - converts files to arrays for both image contents and labels using numpy
write - write idx data to file in given out dir using idx2numpy
"""

from PIL import Image
import idx2numpy as idx
import numpy as np
import argparse
import os
from math import sqrt


def main():
    """ :return: None

    Declare all arguments using argsparse, run through some basic input checks,
    then compile and write idx files if args.run is true; else print dataset size and dimensions.
    """
    parser = argparse.ArgumentParser("parse jpgs and tags into single idx dataset")
    parser.add_argument('--in_dir',  help="pass directory to be processed into idx dataset, default: cwd")
    parser.add_argument('--out_dir', help="directory where set will be written to, default: cwd")
    parser.add_argument('--dict_string', nargs='*',
                        help='list of strings that correspond the tag to the file name, i.e ["a=[1,0,0,0]",]')
    parser.add_argument('--setname', type=str,  help='name of exported dataset')
    parser.add_argument('--run', action='store_true',
                        help='exports the dataset, by default set_compiler will print all data with out export')
    parser.add_argument('--dimensions', help='set the dimensions of the dataset: default is the sqrt() of the pixels')
    args = parser.parse_args()

    parse_list = ['jpg']

    if not args.in_dir:
        args.in_dir = os.getcwd()

    if not args.out_dir:
        args.out_dir = os.getcwd()

    data_matrix = compile_set(args, parse_list)

    if args.run:
        idx_export(args, data_matrix[0], '_labels.idx')         # export idx of labels
        idx_export(args, data_matrix[1], '_images.idx')         # export idx of images
    else:
        print 'number of examples:', data_matrix[1][0].astype(int)
        print 'Dimensions:', data_matrix[1][1:3].astype(int)


def compile_set(args, parse_list):
    """ Iterate through the set of given files and generate a numpy matrix for every image.
    Iterate through all jpg file names and generate label according to the given convention.
    Insert matrix info to the matrices and return list of the two sets

    :param args:
    :param parse_list: all jpgs to be converted and compiled onto the idx format
    :return: np.array of labels and images in a list of the two sets
    """
    file_list = os.listdir(args.in_dir)
    x_list = []
    y_list= []
    n_count = 0
    counter = 0

    for image in file_list:         # generate image matrix
        if image.split('.')[-1] in parse_list:
            x_list[0:0] = [i for i in np.array(Image.open(image).getdata())]
            n_count += 1

            if not args.dimensions:         # set dimension to sqrt() of pixels
                sqr_dim = sqrt(len(np.array(Image.open(image).getdata())))
                args.dimensions = [sqr_dim, sqr_dim]
                counter += 1

            for i in range(len(args.dict_string)):              # generate label matrix
                if args.dict_string[i].split('=')[0] in image:
                    y_list.insert(0, np.array([x for x in args.dict_string[i].split('=')[1]]).astype(float))

    x_list[0:0] = args.dimensions
    x_list.insert(0, n_count)
    data_matrix = [np.array([i for i in y_list]),
                   np.array([i for i in x_list]).astype(float)]  # list of [np.array(labels), np.array(images)]

    return data_matrix


def idx_export(args, data_matrix, ext):
    """Export given matrix out in idx format to args.out_dir"""
    path = args.out_dir+'/'+args.setname+ext
    idx.convert_to_file(path, data_matrix)


if __name__ == '__main__':
    main()
