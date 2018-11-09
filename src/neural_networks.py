import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from lib.sutils import SpatialNonlinearityLayer, SpatialReshapeLayer
from lib.crfrnn.layers import CRFasRNNLayer

def build_network(networkType, input_var, num_classes, num_filters=64,
                  nonlin=lasagne.nonlinearities.rectify, noise=0.0, # Would leaky_rectify improve network?
                  W_init=lasagne.init.GlorotUniform(),
                  b_init=lasagne.init.Constant(0.01), **kwargs):
    if networkType == 'uunet':
        return build_uunet_network(networkType, input_var, num_classes, num_filters,
                          nonlin, noise, # Would leaky_rectify improve network?
                          W_init,b_init, **kwargs)
    else:
        return None, None, None, None

def build_uunet_network(network, input_var, num_classes, num_filters=64,
                  nonlin=lasagne.nonlinearities.rectify, noise=0.0, # Would leaky_rectify improve network?
                  W_init=lasagne.init.GlorotUniform(),
                  b_init=lasagne.init.Constant(0.01), **kwargs):
    """Builds a fully-convolutional U-Net model."""

    crop_mode = [None, None, 'center', 'center']
    image_mean = np.array([103.939, 116.779, 123.68]).reshape(1, 3, 1, 1)

    # To use VGG weights, change RGB to BGR and subtract the mean BGR values
    sym_var = input_var[:, ::-1] - image_mean.astype(theano.config.floatX)

    input_layer = nn.InputLayer((None, 3, None, None), sym_var)

    network = nn.Conv2DLayer(input_layer, 64, 3, pad=1, flip_filters=False)
    level1  = nn.Conv2DLayer(network, 64, 3, pad=1, flip_filters=False)
    network = nn.Pool2DLayer(level1,  2)

    network = nn.Conv2DLayer(network, 128, 3, pad=1, flip_filters=False)
    level2  = nn.Conv2DLayer(network, 128, 3, pad=1, flip_filters=False)
    network = nn.Pool2DLayer(level2,  2)

    network = nn.Conv2DLayer(network, 256, 3, pad=1, flip_filters=False)
    network = nn.Conv2DLayer(network, 256, 3, pad=1, flip_filters=False)
    level3  = nn.Conv2DLayer(network, 256, 3, pad=1, flip_filters=False)
    network = nn.Pool2DLayer(level3,  2)

    network = nn.Conv2DLayer(network, 512, 3, pad=1, flip_filters=False)
    network = nn.Conv2DLayer(network, 512, 3, pad=1, flip_filters=False)
    level4  = nn.Conv2DLayer(network, 512, 3, pad=1, flip_filters=False)
    network = nn.Pool2DLayer(level4,  2)

    network = nn.Conv2DLayer(network, 512, 3, pad=1, flip_filters=False)
    network = nn.Conv2DLayer(network, 512, 3, pad=1, flip_filters=False)
    level5  = nn.Conv2DLayer(network, 512, 3, pad=1, flip_filters=False)
    network = nn.Pool2DLayer(level5,  2)

    # Set the weights for VGG portion
    vgg_layers = network

    # Decoder phase, all these weights will be learned
    network = nn.batch_norm(nn.Conv2DLayer(network, 1024, 1, pad='same'))
    network = nn.batch_norm(nn.Conv2DLayer(network, 1024, 1, pad='same'))
    network = nn.TransposedConv2DLayer(network, 512, 3, stride=2, crop='valid')
    network = nn.batch_norm(network)
    network = nn.ConcatLayer([network, level5], cropping=crop_mode)

    network = nn.batch_norm(nn.Conv2DLayer(network, 512, 3, pad='same'))
    network = nn.batch_norm(nn.Conv2DLayer(network, 521, 3, pad='same'))
    network = nn.TransposedConv2DLayer(network, 512, 3, stride=2, crop='valid')
    network = nn.batch_norm(network)
    network = nn.ConcatLayer([network, level4], cropping=crop_mode)

    # @two convolutions of the same size with a max pooling step in the middle. How is this useful?
    network = nn.batch_norm(nn.Conv2DLayer(network, 512, 3, pad='same'))
    network = nn.batch_norm(nn.Conv2DLayer(network, 521, 3, pad='same'))
    network = nn.TransposedConv2DLayer(network, 256, 3, stride=2, crop='valid')
    network = nn.batch_norm(network)
    network = nn.ConcatLayer([network, level3], cropping=crop_mode)

    network = nn.batch_norm(nn.Conv2DLayer(network, 256, 3, pad='same'))
    network = nn.batch_norm(nn.Conv2DLayer(network, 256, 3, pad='same'))
    network = nn.TransposedConv2DLayer(network, 128, 3, stride=2, crop='valid')
    network = nn.batch_norm(network)
    network = nn.ConcatLayer([network, level2], cropping=crop_mode)

    network = nn.batch_norm(nn.Conv2DLayer(network, 128, 3, pad='same'))
    network = nn.batch_norm(nn.Conv2DLayer(network, 128, 3, pad='same'))
    network = nn.TransposedConv2DLayer(network, 64, 3, stride=2, crop='valid')
    network = nn.batch_norm(network)
    network = nn.ConcatLayer([network, level1], cropping=crop_mode)

    # Final few valid convolutions to output class predictions
    network = nn.batch_norm(nn.Conv2DLayer(network, 64, 3, pad='same'))
    network = nn.Conv2DLayer(network, num_classes, 1, pad='same', nonlinearity=None)
    softmax = SpatialNonlinearityLayer(network, lasagne.nonlinearities.softmax)

    # This reshapes previous layer from 2d [batch_size, num_channels * rows * cols]
    # to [batch_size, num_channels, rows, cols]
    target_shape = (sym_var.shape[0], num_classes,
                    sym_var.shape[2], sym_var.shape[3])
    output = SpatialReshapeLayer(softmax, target_shape)

    # Applies a CRF to the output predictions & cleans everything up
    output_crf = CRFasRNNLayer(output, input_layer, normalize_final_iter=True)

    return softmax, output, output_crf, vgg_layers
