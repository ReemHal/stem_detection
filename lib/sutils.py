import os
import numpy as np
import theano
import lasagne
import lasagne.layers as nn

import cv2
import seaborn as sns

from PIL import Image

def color_map(num=256, normalized=False):
    """
    See: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((num, 3), dtype=dtype)
    for i in range(num):
        red = green = blue = 0
        c = i
        for j in range(8):
            red = red | (bitget(c, 0) << 7-j)
            green = green | (bitget(c, 1) << 7-j)
            blue = blue | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([red, green, blue])

    cmap = cmap/255 if normalized else cmap
    return cmap


def map_pixels(mask, cmap=None):
    """Converts each class-predicted pixel value in an image to an RGB tuple."""

    if cmap is not None:
        palette = cmap
    else:
        palette = sns.color_palette('hls', 12)

    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in xrange(mask.shape[0]):
        for j in xrange(mask.shape[1]):
            rgb = palette[mask[i, j]]
            output[i, j] = (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)
    return output


class SpatialNonlinearityLayer(nn.base.Layer):
    """Applies a nonlinearity across the channels of an image."""
    def __init__(self, incoming, nonlinearity=None, **kwargs):
        super(SpatialNonlinearityLayer, self).__init__(incoming, **kwargs)

        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        return (None, input_shape[1])

    def get_output_for(self, input, **kwargs):
        """Returns spatial softmax across channels of a 4d image tensor.

        As the theano softmax requires a 2d input, we need to
        do a little dimshuffling and reshaping to get the right format.
        """

        incoming = input.dimshuffle(0, 2, 3, 1) # (b, h, w, c)
        incoming = incoming.reshape((-1, incoming.shape[3])) # (b * h * w, c)

        return self.nonlinearity(incoming)


class SpatialReshapeLayer(nn.base.Layer):
    """Implements spatia-softmax across the output channels."""
    def __init__(self, incoming, target_shape, **kwargs):
        super(SpatialReshapeLayer, self).__init__(incoming, **kwargs)

        if isinstance(target_shape, tuple):
            target_shape = (-1, target_shape[1], target_shape[2], target_shape[3])
        self.target_shape = target_shape # Can be a theano symbolic shape

    def get_output_shape_for(self, input_shape):
        return (None, self.target_shape[1], None, None)

    def get_output_for(self, input, **kwargs):
        """Returns a permuted view of a given input.

        First reshape to (batch, rows, cols, channels), then perform a
        dimshuffle to return a standardized (batch, channels, rows, cols) view.
        """

        incoming = input.reshape((self.target_shape[0], self.target_shape[2],
                                  self.target_shape[3], self.target_shape[1]))
        return incoming.dimshuffle(0, 3, 1, 2)


def get_tomato_stats(tomato_contours, kernel_size=3, area_thresh=500):
    """Returns a dictionary of stats for predicted tomato segmentations."""

    # Since the watershed draws contours, we need to invert the predictions to
    # get the 'inside' blob portion. We also slightly compress the blob portion
    # so we can get a more defining border.
    inverted_contours = 255 - tomato_contours

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    inverted_contours = cv2.erode(inverted_contours, kernel, iterations=1)

    # Create a label for each component in the image
    nb_components, output, bbox, centroids = \
        cv2.connectedComponentsWithStats(inverted_contours, connectivity=8)

    # Only keep blobs that are above a certain area threshold. Zero out all
    # blobs that are below this threshold
    good_indices = []
    for i, row in enumerate(bbox):
        if 10000 > row[-1] > area_thresh:
            good_indices.append(i)

    # if no blobs / tomatoes were found
    if len(good_indices) == 0:
        return None, None
    good_indices = np.hstack(good_indices)

    # Zero out all detections on the image
    for i in np.arange(nb_components):
        if i not in good_indices:
            output[output == i] = 0

    stats = {}
    stats['count'] = len(good_indices)
    stats['bbox'] = bbox[good_indices] #(tl_x, tl_y, width, height, area)
    stats['centroid'] = centroids[good_indices].astype(np.int32)
    return stats, output


def get_watershed(segmentation, background, dist_thresh=0.6):
    """Creates a tomato segmentation mask given segmentations from VGG network."""

    # Use a distance transform to find the seed points for watershed
    dist = cv2.distanceTransform(segmentation, cv2.DIST_L2, 5)

    # Since there may be multiple peaks, we use dilation to find them
    kernel = np.ones((5, 5), np.float32)
    dilate = cv2.dilate(dist, kernel, iterations=3)
    where = np.where(dilate == dist, np.ones(dist.shape), np.zeros(dist.shape))
    where = cv2.dilate(where, kernel, iterations=2)
    foreground = np.uint8(where * segmentation)

    # Add one to all labels so that known background is not 0, but 1
    _, markers = cv2.connectedComponents(foreground)
    markers = markers + 1

    unknown = background - foreground
    markers[unknown == 255] = 0

    # Use the distance transformation as the image watershed will use,
    # along with initial seed points
    dist3d = dist * 255
    dist3d = np.tile(dist3d[:, :, np.newaxis], (1, 1, 3)).astype(np.uint8)

    markers = cv2.watershed(dist3d, markers)

    segmentation = np.zeros(segmentation.shape, dtype=np.uint8)
    segmentation[markers == -1] = 255
    return segmentation


def merge_segmented_tomatoes(segmentation, stats, dist_thresh=40, area_thresh=400):

    # Convert values to [0, 255] then calculate hough circles
    gray = segmentation.astype(np.uint8)
    gray[gray > 0] = 255

    merged_indices, new_centroid, new_bbox = [], [], []

    # Calculate distances between all contours and all other contours
    # If distance between two very small contours is low, merge them
    diff = stats['centroid'][np.newaxis] - stats['centroid'][:, np.newaxis]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    area_mask = np.atleast_2d(stats['bbox'][:, -1] < area_thresh)
    dist_mask = np.where(distances < dist_thresh, 1, 0)
    area_dist_mask = dist_mask * area_mask

    # Find the rows with maximum number of 1's, and process those first
    sorted_rows = np.argsort(np.sum(area_dist_mask, axis=1))[::-1]

    for row in sorted_rows:

        # Find other tomatoes that fall in the same mask
        where = np.argwhere(area_dist_mask[row] == 1).flatten()
        if len(where) > 1:

            merged_indices.extend(where)

            indices = np.hstack(where)
            centroid = np.mean(stats['centroid'][indices], axis=0)
            centroid = centroid.astype(np.int32)
            new_centroid.append(np.atleast_2d(centroid))

            bbox = np.zeros((1, 5), dtype=np.int32)
            bbox[0, 0] = np.min(stats['bbox'][indices, 0])
            bbox[0, 1] = np.min(stats['bbox'][indices, 1])
            bbox[0, 2] = np.max(stats['bbox'][indices, 2])
            bbox[0, 3] = np.max(stats['bbox'][indices, 3])
            bbox[0, 4] = np.max(stats['bbox'][indices, 4])
            new_bbox.append(bbox)

            area_dist_mask[:, indices] = 0


    # Keep track of all the detections that we didn't merge
    unmerged_indices = np.arange(len(stats['centroid']))
    unmerged_indices = [u for u in unmerged_indices if u not in merged_indices]
    unmerged_indices = np.hstack(unmerged_indices)

    for u in unmerged_indices:
        new_bbox.append(np.atleast_2d(stats['bbox'][u]))
        new_centroid.append(np.atleast_2d(stats['centroid'][u]))

    # Return everything
    new_stats = {}
    new_stats['centroid'] = np.vstack(new_centroid)
    new_stats['bbox'] = np.vstack(new_bbox)
    return new_stats


def plot_segmentations(inputs, preds, targets=None, prefix=None, save_dir='samples'):
    """Plots an RGB image of [RGB Original, Predicted, Ground Truth] images."""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if inputs.ndim != 4:
        raise Exception('Input image must have 4 channels (bc01)')
    n_samples, _, n_rows, n_cols = inputs.shape

    if preds.ndim == 4 and preds.shape[1] == 1:
        argmax_preds = preds
    elif preds.ndim == 2:
        argmax_preds = preds[np.newaxis, np.newaxis]
    else:
        argmax_preds = np.argmax(preds, axis=1)[:, np.newaxis]

    # How many images (wide) we'll plot
    n_images = 2 if targets is None else 3

    for i in xrange(n_samples):

        # Each output image is composed of 3 equally sized images
        image_shape = (n_rows, n_cols * n_images, 3)
        composite = np.zeros(image_shape, dtype=np.uint8)

        # Flips the channels to be the last dimension
        image = inputs[i].transpose(1, 2, 0).astype(np.float32)
        image = (image / np.max(image)) * 255
        composite[:, :n_cols] = image.astype(np.uint8)
        composite[:, n_cols:2*n_cols] = map_pixels(argmax_preds[i, 0])

        if targets is not None:
            composite[:, 2*n_cols:] = map_pixels(targets[i])

        im = Image.fromarray(composite)

        if prefix is None:
            im.save(os.path.join(save_dir, '%i.jpeg'%i))
        else:
            im.save(os.path.join(save_dir, str(prefix) + '%i.jpeg'%i))


def overlay_predictions(inputs, preds, targets, classes_to_plot, save_dir):
    """Overlays predicted and target segmentations over an input RGB image."""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_samples, _, n_rows, n_cols = inputs.shape

    # Whether the predictions are densely labeled, or a unique channel per class
    if preds.shape[1] == 1:
        argmax_preds = preds
    else:
        argmax_preds = np.argmax(preds, axis=1)[:, np.newaxis]

    if not isinstance(classes_to_plot, list):
        classes_to_plot = [classes_to_plot]

    for i in xrange(n_samples):

        # Image is composed of 2 labeled images: predictions and targets
        image_shape = (n_rows, n_cols * 2, 3)
        composite = np.zeros(image_shape, dtype=np.uint8)

        # Scale the RGB image so it only contributes half of the overall info
        image = inputs[i].transpose(1, 2, 0) * 127
        image = image.astype(np.uint8)

        # This allows us to control which classes we want to plot, by checking
        # whether or not each pixel exists in the classes_to_plot list.
        # Note: This is probably super inefficient.
        max_class = np.max(targets)
        for j in xrange(inputs.shape[2]):
            for k in xrange(inputs.shape[3]):
                if argmax_preds[i, 0, j, k] not in classes_to_plot:
                    argmax_preds[i, 0, j, k] = max_class
                if targets[i, j, k] not in classes_to_plot:
                    targets[i, j, k] = max_class

        # Map each pixel in the predict + target images to an RGB value,
        # which depends on the corresponding class
        cmap = color_map(10)
        image_tf = (map_pixels(argmax_preds[i, 0], cmap) * 0.5).astype(np.uint8)
        composite[:, :n_cols] = image + image_tf
        composite[:, n_cols:2*n_cols] = image + map_pixels(targets[i, 0], cmap) *0.5

        im = Image.fromarray(composite)
        im.save(os.path.join(save_dir, 'overlay%i.jpeg'%i))


def meanIOU(preds, targets, num_classes):
    """Returns the mean intersection over union.

    Notes
    -----
    num_classes is used to create a fixed-sized array, and is in the instance
    where an image might not contain the full set of output classes
    """

    unique_classes = np.unique(targets)

    if preds.ndim == 4:
        mask = np.argmax(preds, axis=1)
    else:
        mask = preds

    # Keep track of an array for each input prediction / target pair.
    # Also keep track of a "found" mask, so when we calculate the mean
    # it takes into account any instances where a class doesn't exist in image
    iou = np.zeros((mask.shape[0], num_classes))
    found = np.zeros((mask.shape[0], num_classes))
    for i in xrange(preds.shape[0]):

        # Calculate the IOU for each class
        for j, cl in enumerate(unique_classes):

            intersection = (mask[i] == cl) * (targets[i] == cl)
            union = (mask[i] == cl) + (targets[i] == cl)

            iou[i, j] = np.sum(intersection > 0) / (np.sum(union > 0) + 1e-8)

            if j in targets[i]:
                found[i, j] = 1.

    return np.mean(iou, axis=0), np.sum(found, axis=0)


def confusion(preds, targets):
    """Outputs a multivariable confusion matrix using predictions + targets."""

    argmax_preds = np.argmax(preds, axis=1)

    yp = argmax_preds.reshape(-1, 1)
    yt = targets.reshape(-1, 1)

    unique = np.unique(yt)
    num_classes = len(unique)
    results = np.zeros((num_classes, num_classes + 2), dtype=np.float32)

    for i in unique:
        print '%8d'%i,
        where = np.where(yt == i)[0]
        for j in unique:
            results[i, j] = np.sum(yp[where] == j)

    # Precision = TP / (TP + FP)
    print "prec.\t",
    tp = np.diag(results)
    results[:, -2] = tp / (np.sum(results[:, :num_classes], axis=0))

    # Recall = TP / (TP + FN)
    print "recall.\t"
    results[:, -1] = tp / (np.sum(results[:, :num_classes], axis=1))

    print '\n'
    for row in results:
        for col in row:
            if col > 1:
                print '%8d'%col,
            else:
                print '%7f'%col,
        print ''


def random_flip(X, Y, p_flip=0.5):
    """Randomly flips pixels across the y-axis with probability p_flip"""

    if X.ndim != 4 or Y.ndim != 3:
        raise Exception('X must be of shape (bc01), Y must be(B01)')
    elif X.shape[0] != Y.shape[0]:
        raise Exception('X and Y must have the same batch size')

    # Loop over each sample and flip mirror those across y with probability p
    for i in xrange(X.shape[0]):
        if np.random.rand < p_flip:
            X[i] = X[i, :, :, ::-1]
            Y[i] = Y[i, :, ::-1]
    return X, Y

def random_crop(X, Y, target_rows=None, target_cols=None):
    """Randomly crops an input X and Y to be size (target_rows, target_cols)."""

    # If no target rows / cols provided, assume we don't actually want to crop
    if target_rows is None and target_cols is None:
        return X, Y
    elif target_rows is None:
        target_rows = X.shape[2]
    else:
        target_cols = X.shape[3]

    row_range = X.shape[2] - target_rows
    col_range = X.shape[3] - target_cols

    assert (X.shape[2] >= target_rows) and (X.shape[3] >= target_cols)

    # Structures for "cropped" versions of images
    cx = np.zeros((X.shape[0], X.shape[1], target_rows, target_cols))
    cy = np.zeros((Y.shape[0], target_rows, target_cols), dtype=np.int32)

    for i in xrange(X.shape[0]):

        # Randomly choose the top-left (x, y) corner
        tlx = np.random.choice(np.arange(col_range)) if col_range > 0 else 0
        tly = np.random.choice(np.arange(row_range)) if row_range > 0 else 0

        # Extract a crop that matches shape (target_rows, target_cols)
        cx[i] = X[i, :, tly: tly + target_rows, tlx: tlx + target_cols]
        cy[i] = Y[i,    tly: tly + target_rows, tlx: tlx + target_cols]

        # Randomly flip / mirror the images
        cx[i], cy[i] = random_flip(cx[i:i+1], cy[i:i+1])

    return cx.astype(X.dtype), cy.astype(Y.dtype)


def load_data(image_file, label_file, seed=1234, n_test=10):
    """Returns a train/test split of a dataset."""

    np.random.seed(seed)

    images = np.load(image_file)['arr_0'].astype(theano.config.floatX)
    labels = np.load(label_file)['arr_0'].astype(np.int32)

    if n_test > images.shape[0]:
        raise Exception('n_test must be less then # %d'%images.shape[0])

    # Labels were encoded as dense masks, where all labels are on a single
    # channel of the image
    num_samples = labels.shape[0]
    print "labels shape:", labels.shape, labels.dtype, np.unique(labels)
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    X_train = images[idx[:-n_test]]
    y_train = labels[idx[:-n_test]]
    X_valid = images[idx[-n_test:]]
    y_valid = labels[idx[-n_test:]]
    print "when loading..labels in train: ", np.unique(y_train), "in test: ", np.unique(y_valid)
    return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    rgb = np.load('output/rgb.npz')['arr_0']
    preds = np.load('output/predictions.npz')['arr_0'].astype(np.float32)


    background_idx = np.unique(preds)[-1]
    tomato_idx = 2

    if preds.shape[1] == 1:
        argmax_preds = preds
    else:
        argmax_preds = np.argmax(preds, axis=1)

    rgb = (rgb * 255).astype(np.uint8)

    # Loop through each sampled image
    for i in xrange(rgb.shape[0]):
        image = rgb[i].transpose(1, 2, 0) # (rows, cols, channels)
        tomatoes = (argmax_preds[i] == tomato_idx).astype(np.uint8)

        background = np.invert(argmax_preds[i] == background_idx)
        background = background.astype(np.uint8) * 255

        seg = get_watershed(tomatoes, background)
        stats, segmented_tomatoes = get_tomato_stats(seg, area_thresh=10)

        if stats is None:
            print 'No tomatoes detected in image.'
            continue

        plt.imshow(segmented_tomatoes)
        plt.show()
