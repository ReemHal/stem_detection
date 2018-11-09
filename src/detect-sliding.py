""" To Fix: This script has improper image directory paths and does not work as is.
    the masks are in the data/composites subdir not data/images.
"""

import sys
sys.path.append('..')
import os
import time
import json
import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers as nn

import matplotlib.pyplot as plt
import cv2
from scipy import misc # Reading images

from lib.dutils import (get_watershed, get_tomato_stats, merge_segmented_tomatoes)

from vggunet import build_network

IMAGE_DIR = ''

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def load_dataset(fname='../data/loc-tomato-labels.json', n_test=10, seed=1234):

    np.random.seed(seed)

    with open(fname) as data_file:
        data = json.load(data_file)

    image_names = data.keys()
    np.random.shuffle(image_names)

    # Split the dataset into training / testing partitions
    return data, image_names[:-n_test], image_names[-n_test:]


def load_image(fname, kernel_size=5, num_iter=5):
    """Loads and processes a single RGB and mask image from disk."""

    rgb_image = misc.imread(fname  + '.jpg')
    mask_image = misc.imread(fname + '_mask.jpg')

    # Close some small holes on the object mask for e.g. if there's missing info
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel,
                                  iterations=num_iter)
    mask_image = mask_image.astype(np.float32) / 255.
    mask_image = 1. - mask_image.round()

    return rgb_image, mask_image[:, :, np.newaxis]


def load_subset(dataset_json, images, patch_size=80):
    """Loads an equal number of classes / non-class image patches from an image"""

    if not isinstance(images, list):
        images = [images]

    rgb_images, bbox = [], []
    for image in images:

        # Many "clusters" of tomatoes are from the same image, so we can save
        # some time not having to load the same images from disk repeatedly
        rgb, _ = load_image(os.path.join(IMAGE_DIR, image))
        rgb = rgb.astype(np.float32) / 255.

        rgb_images.append(rgb.transpose(2, 0, 1))
        bbox.append(np.vstack(dataset_json[image]['bbox']))

    return rgb_images, bbox


def get_crops(image, bbox, offset=5, equal_rate=True):

    assert image.ndim == 3, 'Image must be an array of (rows, cols, channels)'

    cx = ((bbox[:, 0] +  bbox[:, 2]) / 2).astype(np.int32)
    cy = ((bbox[:, 1] +  bbox[:, 3]) / 2).astype(np.int32)
    centroids = np.hstack([cx, cy])

    def random_crop(input, width, height):

        while True:
            y = np.random.randint(0, input.shape[1] - height)
            x = np.random.randint(0, input.shape[2] - width)

            c = np.hstack([y, x])
            if all(np.sqrt(np.sum((c - ct) ** 2)) > 10 for ct in centroids):
                break
        return input[:, y:y+height, x:x+height]


    positive_crops, negative_crops = [], []
    for (x, y, w, h) in bbox:

        y = np.maximum(0, y - offset)
        x = np.maximum(0, x - offset)
        h = np.minimum(image.shape[1], h + offset)
        w = np.minimum(image.shape[2], w + offset)

        positive_crops.append(image[:, y:y+h, x:x+w])
        negative_crops.append(random_crop(image, height=50, width=50))

    return positive_crops, negative_crops


def iterate_minibatches(inputs, bbox, batchsize=1, shuffle=False):


    assert len(inputs) == len(bbox)

    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

        excerpt = indices[start_idx]

        positive, negative = get_crops(inputs[excerpt], bbox[excerpt])

        for pos, neg in zip(positive, negative):
            yield pos, 1
            yield neg, 0


def build_network(input_var=None, input_shape=227):

    nf = 32
    n = lasagne.nonlinearities.tanh
    W_init = lasagne.init.GlorotUniform()


    net = nn.InputLayer((None, 3, None, None), input_var=input_var)

    # Block 1
    net = nn.Conv2DLayer(net, nf, 3, W=W_init, nonlinearity=n, pad='same')
    net = nn.Conv2DLayer(net, nf, 3, W=W_init, nonlinearity=n, pad='same')
    net = nn.MaxPool2DLayer(net, 2)

    # Block 2
    net = nn.Conv2DLayer(net, nf*2, 3, W=W_init, nonlinearity=n, pad='same')
    net = nn.Conv2DLayer(net, nf*2, 3, W=W_init, nonlinearity=n, pad='same')
    net = nn.SpatialPyramidPoolingLayer(net, [4, 2, 1], implementation='kaiming')

    net = nn.DenseLayer(net, 512, W=W_init, nonlinearity=n)
    net = nn.dropout(net, p=0.5)

    net = nn.DenseLayer(net, 128, W=W_init, nonlinearity=n)

    return nn.DenseLayer(net, 1, W=W_init, nonlinearity=T.nnet.sigmoid)


def test_detector(param_file='tomato-model.npz'):

    seed = 1234
    win = 60
    step = 1
    input_var = T.tensor4('x')

    # Load images and network weights
    fname = '../data/loc-tomato-labels.json'
    dataset_json, train_keys, valid_keys = load_dataset(fname, n_test=10, seed=seed)

    train_images, train_bbox = load_subset(dataset_json, train_keys)
    valid_images, valid_bbox = load_subset(dataset_json, valid_keys)

    network = build_network(input_var)
    with np.load(param_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Compile a function for getting output from the network
    output = nn.get_output(network, deterministic=True)

    get_output = theano.function([input_var], output)

    for image in train_images:

        canvas = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)

        im = np.pad(image, ((0, 0), (win // 2, win // 2),
                            (win // 2, win // 2)), mode='reflect')

        for i in range(0, im.shape[1] - win, step):
            for j in range(0, im.shape[2] - win, step):

                patch = im[:, i:i+win, j:j+win][np.newaxis]
                p = get_output(patch)[0, 0]
                canvas[i, j] = p

            if i % int(im.shape[1] / 10.) == 0:
                print i * 10, ' percent done.'

        import matplotlib.pyplot as plt
        canvas /= np.max(canvas)
        canvas = canvas * 255
        canvas = canvas.astype(np.uint8)
        plt.imshow(canvas)
        plt.grid('off')
        plt.show()



def main(num_epochs=500, seed=1234, tomato_idx=2, background_idx=7):

    test_detector()
    sys.exit(1)


    lrate = 1e-5

    fname = '../data/loc-tomato-labels.json'
    dataset_json, train_keys, valid_keys = load_dataset(fname, n_test=10, seed=seed)

    train_images, train_bbox = load_subset(dataset_json, train_keys)
    valid_images, valid_bbox = load_subset(dataset_json, valid_keys)


    # Prepare Theano variables for inputs and targets
    print('Building network ...')
    input_var = T.tensor4('inputs')
    target_var = T.scalar('targets')

    network = build_network(input_var)

    #with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #lasagne.layers.set_all_param_values(softmax, param_values)

    # Create a loss expression for training
    output = nn.get_output(network, deterministic=False)
    loss = T.mean(lasagne.objectives.binary_crossentropy(output, target_var))

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=lrate, rho=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)


    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    output = nn.get_output(network, deterministic=True)
    loss = T.mean(lasagne.objectives.binary_crossentropy(output, target_var))

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [loss, output])



    # Finally, launch the training loop.
    print("Starting training...")
    best_err = np.inf

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_images, train_bbox, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs[np.newaxis], targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        val_preds = []
        val_tgts = []
        for batch in iterate_minibatches(valid_images, valid_bbox, shuffle=False):
            inputs, targets = batch
            count_err, pred = val_fn(inputs[np.newaxis], targets)
            val_err += count_err
            val_batches += 1
            val_preds.append(pred[0])
            val_tgts.append(targets)


        val_preds = np.atleast_2d(np.hstack(val_preds)).T
        val_tgts = np.atleast_2d(np.hstack(val_tgts)).T
        acc = np.mean(val_preds.round() == val_tgts)
        #print np.hstack([val_preds, val_tgts])

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
                        epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation acc: \t\t{:.6f}".format(acc))


       # Early stopping
        if val_err > best_err*0.99:
            count += 1
        else:
            count = 0
            best_err = val_err
            best_params = nn.get_all_param_values(network)
        if count >= 6:
            nn.set_all_param_values(network, best_params)
            break

    np.savez('tomato-model.npz', *lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    main()
