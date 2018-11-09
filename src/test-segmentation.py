import os
import sys
sys.path.append('..')
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from scipy import misc
from vggunet import build_network
from lib.sutils import plot_segmentations


def build_segmentation_function(param_file='../data/trained-weights.npz'):

    input_var = T.tensor4('x')
    num_classes = 8

    softmax, net, net_crf, _ = build_network(input_var, num_classes)

    # Initialize with pretrained weights
    with np.load(param_file) as f:
        param_values = [f['arr_%d' % i] for i in xrange(len(f.files))]
    nn.set_all_param_values(softmax, param_values)

    # Get the output of the networks. 
    output, output_crf = nn.get_output([net, net_crf], deterministic=True)

    # Process info for network path with & without CRF
    shape = (output.shape[2], output.shape[3])

    output = T.argmax(output, axis=1).reshape(shape)
    output_crf = T.argmax(output_crf, axis=1).reshape(shape)

    # Compile the function
    function = theano.function([input_var], [output, output_crf], 
                               allow_input_downcast=True)  
    
    return function 


if __name__ == '__main__':

    # Need to load the pretrained weights
    param_file = '../data/trained-weights.npz'
    function = build_segmentation_function(param_file)

    # Read in as (rows, cols, channels), but we need (1, channels, rows, cols)
    rgb = misc.imread('image1075.jpg')
    rgb = rgb[np.newaxis].transpose(0, 3, 1, 2)

    output, crf_output = function(rgb)

    plot_segmentations(rgb, output, prefix='reg', save_dir='../output/sample-seg')
    plot_segmentations(rgb, crf_output, prefix='crf', save_dir='../output/sample-seg')
