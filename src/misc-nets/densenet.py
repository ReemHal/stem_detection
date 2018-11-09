"""
TODO: 
- Remove duplicate images / labels from dataset
- Dataset augmentation
- Mirror images so not padding with 0's
- Find a better way to choose initial crop size of images.
- Standardize images to 0 mean unit variance (batch normalization appears to do
something in this scenario. Maybe lighting / variance too big otherwise?)
"""

import os, sys
import time
from datetime import datetime
import numpy as  np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn
from lasagne.regularization import regularize_network_params, l2

from collections import namedtuple

from utils import *

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_block(incoming, num_layers, num_filters, use_linear_skip=True,
                filter_size=3, p=0.1, 
                W_init = lasagne.init.GlorotUniform(), b_init=None,
                nonlinearity=lasagne.nonlinearities.rectify,):
    """Builds a block in the DenseNet model."""

    feature_maps = [incoming]

    for i in xrange(num_layers):

        if len(feature_maps) == 1:
            network = incoming
        else:
            network = nn.ConcatLayer(feature_maps, axis=1)

        network = nn.BatchNormLayer(network)
        network = nn.NonlinearityLayer(network, nonlinearity)
        network = nn.Conv2DLayer(network, num_filters, filter_size, 
                                 pad='same', W=W_init, b=b_init)
        if p > 0:
            network = nn.DropoutLayer(network, p=p)
        feature_maps.append(network)

    # Whether to return all connections (vanilla DenseNet), or to return only
    # those feature maps created in the current block used in upscale path for 
    # semantic segmentation (100 layer tiramisu)
    if use_linear_skip:
        return nn.ConcatLayer(feature_maps, axis=1)
    return nn.ConcatLayer(feature_maps[1:], axis=1)


def build_transition_up(incoming, incoming_skip, layers_per_block, growth_rate,
                        W_init=lasagne.init.GlorotUniform(), b_init=None):
    """"Builds a transition in the DenseNet model. 

    Transitions consist of the sequence: Batch Normalization, 1x1 Convolution,
    2x2 Average Pooling. The channels can be compressed by specifying 
    0 < m <= 1, where num_channels = channels * m.
    """
    network = nn.TransposedConv2DLayer(incoming, growth_rate * layers_per_block,
                                       filter_size=3, stride=2, crop='valid',
                                       W=W_init, b=b_init)
    cropping = [None, None, 'center', 'center']
    return nn.ConcatLayer([network, incoming_skip], cropping=cropping)


def build_transition_down(incoming, reduction, p=0.1, 
                          W_init=lasagne.init.GlorotUniform(), b_init=None):
    """"Builds a transition in the DenseNet model. 

    Transitions consist of the sequence: Batch Normalization, 1x1 Convolution,
    2x2 Average Pooling. The channels can be compressed by specifying 
    0 < m <= 1, where num_channels = channels * m.
    """
    num_filters = int(incoming.output_shape[1] * reduction)

    network = nn.BatchNormLayer(incoming)
    network = nn.NonlinearityLayer(network, lasagne.nonlinearities.rectify)
    network = nn.Conv2DLayer(network, num_filters, 1, W=W_init, b=b_init)
    if p > 0:
        network = nn.DropoutLayer(network, p=p)
    return nn.Pool2DLayer(network, 2, 2, mode='max')


def build_network(input_var, input_shape, n_classes, layers_per_block=4,
                  growth_rate=16, filter_size=3, p=0.1,
                  W_init=lasagne.init.GlorotUniform(), b_init=None,
                  nonlinearity=lasagne.nonlinearities.rectify):
    """Builds a fully-connected densenet for semantic segmentation."""

    network = nn.InputLayer(input_shape, input_var)

    network = nn.Conv2DLayer(network, num_filters=48, filter_size=3, W=W_init,
                             b=b_init, pad='same')

    # Downscale
    level1 = build_block(network, layers_per_block, growth_rate, p=p)

    network = build_transition_down(level1, 1.0, p=p)

    level2 = build_block(network, layers_per_block, growth_rate, p=p)

    network = build_transition_down(level2, 1.0, p=p)

    level3 = build_block(network, layers_per_block, growth_rate, p=p)

    network = build_transition_down(level3, 1.0, p=p)

    level4 = build_block(network, layers_per_block, growth_rate, p=p)

    network = build_transition_down(level4, 1.0, p=p)

    # Bottleneck
    network = build_block(network, layers_per_block, growth_rate, False, p=p)

    # Upscale
    network = build_transition_up(network, level4, layers_per_block, growth_rate)

    network = build_block(network, layers_per_block, growth_rate, False, p=p)

    network = build_transition_up(network, level3, layers_per_block, growth_rate)

    network = build_block(network, layers_per_block, growth_rate, False, p=p)

    network = build_transition_up(network, level2, layers_per_block, growth_rate)

    network = build_block(network, layers_per_block, growth_rate, False, p=p)

    network = build_transition_up(network, level1, layers_per_block, growth_rate)

    network = build_block(network, layers_per_block, growth_rate, False, p=p)

    network = nn.NonlinearityLayer(network, nonlinearity)

    network = nn.Conv2DLayer(network, num_filters=n_classes, filter_size=1,
                             nonlinearity=None, W=W_init, b=b_init)

    softmax = SpatialSoftmaxLayer(network)

    reshape = SpatialReshapeLayer(softmax, network.output_shape)
    return softmax, reshape

def main():

    # Where we'll save data to
    fname = sys.argv[0].split('.py')[0]
    curr_time = datetime.now().strftime('%d%H%M')
    save_dir = 'sample-' + fname + curr_time


    lrate = 5e-4
    batch_size = 1
    num_epochs = 100
    crop_size = 360
    input_var = T.tensor4('x')
    target_var = T.itensor4('y')


    images = np.load('images.npz')['arr_0'].astype(theano.config.floatX) / 255.0
    labels = np.load('labels.npz')['arr_0'].astype(np.int32)

    num_classes = labels.shape[1]

    idx = np.arange(num_classes)
    idx = idx.reshape(1, num_classes, 1, 1)
    labels = labels / 255
    labels = labels.astype(np.int32) * idx
    labels = np.sum(labels, axis=1, keepdims=True)

    np.random.seed(1234)
    idx = np.arange(images.shape[0])
    np.random.shuffle(idx)
    X_train = images[idx[:-10]]
    y_train = labels[idx[:-10]]
    X_valid = images[idx[-10:]]
    y_valid = labels[idx[-10:]]


    # Compute class weights to balance dataset
    counts = []
    for cl in xrange(num_classes):
        class_counts = 0
        for img in y_train:
            class_counts += np.sum(img == cl)
        counts.append(class_counts)
    counts = np.array(counts).astype(theano.config.floatX)

    # We can either upscale the loss (i.e. multiply by a factor > 1), or 
    # downscale the loss (multiply by a factor < 1). Here we do the latter
    counts = np.max(counts) / counts
    counts = counts / np.max(counts)
    counts[0] = counts[0] * 1.1 # stem
    counts[1] = counts[1] * 1.1 # tomato
    counts = T.as_tensor_variable(counts)



    # Build DenseNetwork
    input_shape = (None, 3, crop_size, crop_size)
    softmax, network = build_network(input_var, input_shape, num_classes)

    print 'Number of paramters: ', nn.count_params(network)

    preds = nn.get_output(softmax, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(preds, target_var.flatten())
    loss = loss * counts[target_var.flatten()]
    loss = T.mean(loss) + regularize_network_params(softmax, l2) * 0.0001


    acc = T.mean(T.eq(T.argmax(preds, axis=1), target_var.flatten()))

    params = nn.get_all_params(softmax, trainable=True)
    updates = lasagne.updates.adam(loss, params, lrate)
    train_fn = theano.function([input_var, target_var], [loss, acc], 
                               updates=updates,
                               allow_input_downcast=True)

    probs, preds = nn.get_output([softmax, network], deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(probs, target_var.flatten())
    loss = loss * counts[target_var.flatten()]
    loss = T.mean(loss) + regularize_network_params(softmax, l2) * 0.0001


    acc = T.mean(T.eq(T.argmax(probs, axis=1), target_var.flatten()))
    
    valid_fn = theano.function([input_var, target_var], [loss, acc, preds],
                               allow_input_downcast=True)


    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0

        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch

            inputs, targets = random_crop(inputs, targets, crop_size, crop_size)

            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1




        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        valid_iou = np.zeros((num_classes, ))
        val_preds, val_inputs, val_targets = [], [], []
        for batch in iterate_minibatches(X_valid, y_valid, batch_size, shuffle=False):
            inputs, targets = batch

            input_crop, target_crop = random_crop(inputs, targets, crop_size, crop_size)

            err, acc, preds = valid_fn(input_crop, target_crop)
            val_err += err
            val_acc += acc
            val_batches += 1
      
            val_preds.append(preds)
            val_inputs.append(input_crop)
            val_targets.append(target_crop)

            valid_iou += meanIOU(preds, target_crop, num_classes)


        if epoch % 2 == 0:
            val_preds = np.vstack(val_preds)
            val_inputs = np.vstack(val_inputs)
            val_targets = np.vstack(val_targets)
            plot_predictions(val_inputs, val_preds, val_targets, epoch, save_dir)

        # Then we print the results for this epoch:
        print "Epoch {} of {} took {:.3f}s".format(
                        epoch + 1, num_epochs, time.time() - start_time)
        print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
        print "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
        print "  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100)
        print "  validation IOU:\t\t{}".format(valid_iou / val_batches)



if __name__ == '__main__':
    main()

