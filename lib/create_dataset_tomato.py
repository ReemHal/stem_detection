import os
import re
import sys
sys.path.append('..')
import glob
import json
from collections import OrderedDict

import cv2
import numpy as np
import seaborn as sns

from scipy import misc # Reading jpg images
from image_annotation_parser import read # Reading annotations
from sutils import map_pixels
import matplotlib.pyplot as plt

# Use this while debigging to see the different masks
SHOWME_FLAG = False

# Where we'll save the dataset, and where the images were downloaded to
GLOBAL_SAVE_DIR = '../data'
GLOBAL_INPUT_DIR = '../data/images'
GLOBAL_COMPOSITE_DIR = os.path.join(GLOBAL_SAVE_DIR, 'composites')
GLOBAL_PALETTE = sns.color_palette("hls", 10)
GLOBAL_IGNORE_LIST = ['image1036', 'image1078', 'image1383']
GLOBAL_SHAPE = (860, 720) # Only want to use images with this default size

# Initialize some data structures for storing images + labels
class_list = ['stem', 'tomato', 'leaf', 'background tomato',
              'background stem', 'background leaf']

GLOBAL_CLASSES = OrderedDict()
for i, label in enumerate(class_list):
    GLOBAL_CLASSES[label] = i

def showme(img, img_title):
    fig=plt.figure().add_subplot(111)
    fig.imshow(img)
    plt.title(img_title)
    plt.show()

def search_image_dir(image_dir, image_shape):
    """Reads the images in a directory and returns the path to those we'll use
    to build a dataset with."""

    # Take a look at all the marked / annotated files.
    # Only keep those images where the (rows, cols) size is equal to the
    # rest of the populaton.
    files = glob.glob(os.path.join(image_dir, '*annotation*.png'))

    good_files = []
    for f in files:

        # There are many duplicate images before 978; Every second image
        # (in order) is a copy. Further, these images have a more coarse
        # segmentation than those later on in the dataset.
        num = f.split('images' + os.path.sep + 'image')[1].split('_')[0]
        if int(num) <= 978 or any(str(num) in a for a in GLOBAL_IGNORE_LIST):
            continue

        data = read(f)

        # Check shape is consistent with all others, and image has been annotated
        if data._image_shape != image_shape:
            continue
        elif len(data._label_sets) == 0:
            continue

        # Record good files and a list of all unique labels we've seen
        good_files.append(f.split('_')[0])
    return good_files


def get_border_masks(object_masks, kernel_size=3):
    """Creates a set of masks indicating the border pixels of object_masks.

    This function uses a succession of erosion and dilation to obtain areas
    around the border of unique objects in object_masks.

    Parameters
    ----------
    object_masks : array of shape (n, rows, cols) where n is the number of
        objects that belong to the same image.
    kernel_size : integer denoting size of the erosion / dilation to apply.
    """

    if object_masks.ndim == 2:
        object_masks = object_masks[np.newaxis]
    if not isinstance(kernel_size, int):
        raise Exception('kernel_size must be an integer value.')

    border_masks = np.zeros_like(object_masks)

    # Loop over all the tomatoes
    for i, mask in enumerate(object_masks):

        # Create a small border around the tomato by performing an erosion and
        # a dilation. The erosion ensures we are able to capture the transition
        # from tomato -> tomato border -> background
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        dilated_image = cv2.dilate(eroded_mask, kernel, iterations=4)

        border_masks[i] = dilated_image - eroded_mask

    return border_masks


def get_object_masks(annotation, class_name):
    """Returns a 3d array, where each channel contains a unique object.

    Parameters
    ----------
    annotation : labeled AnnotationData instance
    class_name : class we wish to index from 'annotation'

    Returns
    -------
    object_masks : array of shape (n, rows, cols), where n is the number of
        [class_name] objects in the image.
    """

    if not isinstance(class_name, str):
        raise Exception('class_name must be of type <str>.')

    image_shape = annotation._image_shape
    num_objects = annotation.get_classes()[class_name]
    object_masks = np.zeros((num_objects, ) + image_shape, np.uint8)

    # Loop over all the tomatoes
    for i in xrange(num_objects):

        mask = annotation.get_mask(class_name, i + 1)

        if mask is None:
            continue
        object_masks[i] = np.uint8(mask * 255)

    return object_masks

def get_stats(tomato_mask, area_thresh=200):
    """Returns a centroid, bounding box and contours for a mask containing a
    unique object.

    Parameters
    ----------
    tomato_mask : array of shape (rows, cols) containing an unique object
    area_thresh : integer specifying the threshold we wish to use for keeping
        a labeled object. Areas less then this threshold will be ignored.
    """

    if not isinstance(area_thresh, int):
        area_thresh = int(area_thresh)

    # Find the object contours independently of the bounding box + centroid
    mask = np.uint8((tomato_mask / np.max(tomato_mask)) * 255)
    if SHOWME_FLAG == True:
        showme(mask.astype(np.uint8), 'Single Object')

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if SHOWME_FLAG == True:
        img = np.zeros(mask.shape)
        cv2.drawContours(img, contours, -1, (255,255,255))
        showme(img, 'Contours')

    for i in xrange(len(contours)):
        contours[i] = contours[i].tolist()

    # For each shape in the image, get the bounding box and centroid. We'll
    # threshold the returned values based on the amount of area the object has
    _, _, bbox, centroid = cv2.connectedComponentsWithStats(mask)
    good_idx = [i for i, row in enumerate(bbox) if 10000 > row[-1] > area_thresh]

    if len(good_idx) == 0:
        return None, None, None
    good_idx = np.hstack(good_idx)

    # If a labeled object is composed of more then one contour or shape, we
    # calculate the shape centroid as the average labeled pixel, and recompute
    # a bounding box that contains all labeled segments.
    if len(good_idx) >= 1:
        centroid = np.mean(centroid[good_idx], axis=0)
        centroid = centroid.astype(int)

        bbox_ = np.zeros((4,), dtype=int)
        bbox_[0] = np.min(bbox[good_idx, 0])
        bbox_[1] = np.min(bbox[good_idx, 1])
        # For the rectangle dimensions sides we need to take the furthest point
        # any of the initial bounding boxes reaches, then use it to calculate the dimensions
        # of the new bbox
        bbox_[2] = np.max([bbox[i,0]+bbox[i,2] for i in good_idx])-bbox_[0]
        bbox_[3] = np.max([bbox[i,1]+bbox[i,3] for i in good_idx])-bbox_[1]
        if SHOWME_FLAG == True: #for debugging purposes
            print "bbox:", bbox_
            cv2.rectangle(img,(bbox_[0],bbox_[1]),(bbox_[0]+bbox_[2],bbox_[1]+bbox_[3]),(255,255,255),2)
            img = img+mask
            print "img type", img.shape
            img[centroid[1]][centroid[0]]=600
            showme(img, 'with bbox')

        return list(centroid.flatten()), list(bbox_.flatten()), contours
    else:
        return [], [], []

    # If a labeled object only contains one contour or shape, we can return
    # a list of centroid and bounding box right away
    bbox = list(bbox[good_idx, :-1].flatten().astype(int))
    centroid = list((centroid[good_idx, :].flatten()).astype(int))

    return centroid, bbox, contours


def get_object_stats(object_masks, area_thresh=200):
    """Finds the object centorid, bounding box, and contours for each object.

    Parameters
    ----------
    object_masks : (n, rows, cols) array, where n is the number of images.
    """

    if object_masks.ndim == 2:
        object_masks = object_masks[np.newaxis]

    # Store information by appending to lists held within a dict
    object_stats = {key : [] for key in ['bbox', 'centroid', 'contours']}

    for mask in object_masks:

        centroid, bbox, contours = get_stats(mask, area_thresh=area_thresh)

        if any(a is None for a in [centroid, bbox, contours]):
            continue

        object_stats['bbox'].append(bbox)
        object_stats['centroid'].append(tuple(centroid))
        object_stats['contours'].append(contours)

    return object_stats


def main():

    all_files = search_image_dir(GLOBAL_INPUT_DIR, GLOBAL_SHAPE)
    if len(all_files) == 0:
        raise Exception('No valid images found in directory %s'%GLOBAL_INPUT_DIR)

    class_name = 'tomato'
    all_images, all_masks, all_stats = [], [], {}

    for i, f in enumerate(all_files):

        # Read the labeled images, and store them in appropriate channel
        annotated = read(glob.glob(f + '_annotation*')[0])

        dense_mask = annotated.get_dense_mask(GLOBAL_CLASSES, background_first=False)

        # In order to augment a dense mask (where classes are stored in
        # priority order), we need to increment the label of all other classes
        # by 1.
        class_name_idx = GLOBAL_CLASSES[class_name]
        dense_mask[dense_mask > class_name_idx] += 1

        # Check whether the object class exists in a given image. If so, we need to
        # carefully construct a 'border class' by iterating through all labels
        if class_name in annotated.get_classes():

            # Create a mask showing all [class_name] objects in the image
            class_mask = annotated.get_mask(class_name)
            if SHOWME_FLAG == True:
                showme(class_mask.astype(np.uint8),'class mask')

            objects = get_object_masks(annotated, class_name)
            borders = get_border_masks(objects, kernel_size=3)

            # [objects] and [borders] are (n, rows, cols) arrays, where n is the
            # number of specific objects in an image. To create a dense mask of
            # these locations, carefully take a summation and remove the
            # intersection between objects and their borders.
            border_mask = np.sum(borders, axis=0)
            border_mask = np.where(border_mask > 0, 1, 0).astype(np.bool)

            # The tomato mask is then the portions of tomato contained in the border
            object_mask = np.bitwise_and(np.invert(border_mask), class_mask)
            object_mask = object_mask.astype(np.uint8)
            border_mask = border_mask.astype(np.uint8)
            if SHOWME_FLAG == True:
                showme(object_mask.astype(np.uint8),'object mask')

            # Replace the [class_name] position in the dense mask with the newly
            # created object and border mask. This requires finding the index of
            # [class_name] in the densely labeled image, putting the new object
            # and border mask in this slot, then incrementing all labels with
            # less priority by 1.
            dense_mask[object_mask > 0] = class_name_idx
            dense_mask[border_mask > 0] = class_name_idx + 1

            all_stats[f] = get_object_stats(objects)

        all_masks.append(dense_mask[np.newaxis])

        # Format each RGB Image to be in the format [batch, channels, rows, cols]
        rgb_image = misc.imread(f + '.jpg')
        rgb_image = rgb_image.transpose(2, 0, 1)[np.newaxis]
        all_images.append(rgb_image)




    # ----------- Save the dataset ------------------------------------

    # [stem, tomato border, tomato, leaf, bg tomato, bg stem, bg leaf, bg]
    all_images = np.vstack(all_images)
    all_masks = np.vstack(all_masks)

    np.savez(os.path.join(GLOBAL_SAVE_DIR, 'seg-tomato-images.npz'), all_images)
    np.savez(os.path.join(GLOBAL_SAVE_DIR, 'seg-tomato-labels.npz'), all_masks)

    json_path = os.path.join(GLOBAL_SAVE_DIR, 'loc-tomato-labels.json')
    with open(json_path, 'w') as outfile:
        json.dump(all_stats, outfile)


    # Update our dictionary mapping each class to a label.
    # Background will always be the final element.
    class_name_idx = GLOBAL_CLASSES[class_name]
    for class_ in GLOBAL_CLASSES:
        if GLOBAL_CLASSES[class_] > class_name_idx:
            GLOBAL_CLASSES[class_] += 1
    GLOBAL_CLASSES[class_name + ' border'] = class_name_idx + 1
    GLOBAL_CLASSES['background'] = max(GLOBAL_CLASSES[c] for c in GLOBAL_CLASSES) + 1

    json_path = os.path.join(GLOBAL_SAVE_DIR, 'segmentation-label-dict.json')
    with open(json_path, 'w') as outfile:
        json.dump(GLOBAL_CLASSES, outfile)


    # ----------- Plot the densely labeled masks ------------------------

    # Plot the segmentation mask and original image on top of each other
    if not os.path.exists(GLOBAL_COMPOSITE_DIR):
        os.makedirs(GLOBAL_COMPOSITE_DIR)

    for i, img in enumerate(all_images):

        print 'Plotting image %s'%all_files[i]
        f = all_files[i].split(os.path.sep)[-1]

        composite = map_pixels(all_masks[i])

        # Overlay the image and the mask
        img = (0.5 * img.transpose(1, 2, 0)).astype(np.uint8)
        overlay = (0.5 * composite).astype(np.uint8)

        fname = os.path.join(GLOBAL_COMPOSITE_DIR, f + '.jpg')
        misc.imsave(fname, img + overlay)

        #for c in all_stats[all_files[i]]['centroid']:
        #    cv2.circle(composite, (c[0], c[1]), 5, (0, 255, 0), -1)

        fname = os.path.join(GLOBAL_COMPOSITE_DIR, f + '_mask.jpg')
        misc.imsave(fname, composite)


if __name__ == '__main__':
    main()
