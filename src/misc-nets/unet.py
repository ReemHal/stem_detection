import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from utils import SpatialNonlinearityLayer, SpatialReshapeLayer


def build_contract_level(incoming, num_filters, nonlin,
                         W_init=lasagne.init.GlorotUniform(),
                         b_init=lasagne.init.Constant(0.01), filter_size=3):
    """Builds a Conv-Conv-Pool block of the U-Net encoder."""

    network = nn.Conv2DLayer(incoming, num_filters, filter_size, pad='same',
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)
    network = nn.Conv2DLayer(network, num_filters, filter_size, pad='same',
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)
    return network, nn.MaxPool2DLayer(network, 2)


def build_expand_level(incoming, incoming_skip, num_filters, nonlin,
                       W_init=lasagne.init.GlorotUniform(),
                       b_init=lasagne.init.Constant(0.01), filter_size=3):
    """Builds a Conv-Conv-Deconv-Concat block of U-Net."""

    network = nn.Conv2DLayer(incoming, num_filters, filter_size, pad='same',
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)
    network = nn.Conv2DLayer(network, num_filters, filter_size, pad='same',
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)
    network = nn.TransposedConv2DLayer(network, num_filters // 2, filter_size,
                                       stride=2, crop='valid', W=W_init,
                                       b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)

    crop_mode = [None, None, 'center', 'center']
    return nn.ConcatLayer([network, incoming_skip], cropping=crop_mode)


def build_network(input_shape, input_var, num_classes, num_filters=32,
                  nonlin=lasagne.nonlinearities.rectify, noise=0.0,
                  W_init=lasagne.init.GlorotUniform(),
                  b_init=lasagne.init.Constant(0.01)):
    """Builds a fully-convolutional U-Net model."""

    input_layer = nn.InputLayer(input_shape, input_var)
    network = input_layer

    if noise > 0:
        network = nn.GaussianNoiseLayer(network, noise)

    level1, network = build_contract_level(network, num_filters*1, nonlin, W_init)

    level2, network = build_contract_level(network, num_filters*2, nonlin, W_init)

    level3, network = build_contract_level(network, num_filters*4, nonlin, W_init)

    level4, network = build_contract_level(network, num_filters*8, nonlin, W_init)

    network = build_expand_level(network, level4, num_filters*16, nonlin, W_init)

    network = build_expand_level(network, level3, num_filters*8, nonlin, W_init)

    network = build_expand_level(network, level2, num_filters*4, nonlin, W_init)

    network = build_expand_level(network, level1, num_filters*2, nonlin, W_init)

    network = nn.Conv2DLayer(network, num_filters, 3, pad='same',
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)

    network = nn.Conv2DLayer(network, num_filters, 3, pad='same',
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.batch_norm(network)
    
    # Final few valid convolutions to output class predictions
    network = nn.Conv2DLayer(network, num_classes, 1, W=W_init, b=b_init)

    softmax = SpatialNonlinearityLayer(network, lasagne.nonlinearities.softmax)

    # Reshape the final layer to look like input images
    var = input_layer.input_var.shape
    target_shape = (var.shape[0], num_classes, var.shape[2], var.shape[3])
    reshape = SpatialReshapeLayer(softmax, target_shape)


    return softmax, reshape, nn.get_all_params(softmax)
