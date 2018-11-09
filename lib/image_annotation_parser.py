#!/usr/bin/env python

import json
import math
import numpy
import struct
from PIL import Image
import numpy as np

class AnnotationData:
    def __init__(self, image_data, metadata):

        self._labels = []

        max_tile = 0
        for label in metadata:
            self._labels.append(label)
            if metadata[label][0][0]+1 > max_tile:
                max_tile = metadata[label][0][0]+1

        self._image_shape = (
                        image_data.shape[0] // max_tile,
                        image_data.shape[1]
                    )

        # Create a dictionary for each label
        # Each dictionary will map object numbers to their data
        self._label_sets = [dict() for i in range(len(self._labels))]

        # Create layer for each label (this will hold all objects on one layer)
        self._label_single = [
                        numpy.zeros(self._image_shape, dtype=bool)
                        for i in range(len(self._labels))
                    ]
        # Create layer for each label (this will hold all objects on one layer,
        # retaining their object numbers)
        self._label_objects = [None for i in range(len(self._labels))]

        ones = numpy.ones(self._image_shape, dtype=numpy.bool)
        zeros = numpy.zeros(self._image_shape, dtype=numpy.bool)

        for label in self._labels:
            index = self._labels.index(label)

            # Take section of image
            offset, layer = metadata[label][0]
            section_start = self._image_shape[0] * offset
            section_end = self._image_shape[0] * (offset + 1)
            image_section = image_data[section_start:section_end, :, layer]

            # Get list of object numbers
            unique_objects = numpy.unique(image_section)

            for obj in unique_objects:
                if obj == 0:
                    continue

                object_mask = numpy.where(image_section == obj, ones, zeros)

                # Ensure data is not overwritten
                if obj in self._label_sets[index]:
                    existing_mask = self._label_sets[index][obj]
                    object_mask = numpy.logical_or(object_mask, existing_mask)

                # Add object to dict
                self._label_sets[index][obj] = object_mask
                # Append object to layer
                self._label_single[index] = numpy.logical_or(
                                        self._label_single[index], object_mask
                                    )

                self._label_objects[index] = image_section

    def get_classes(self):
        """Returns a count of objects in each class

        :return: A dictionary eg. {'tomato': 5, 'leaf': 3, 'stem': 4}
        """
        dictionary = dict()
        for i, label in enumerate(self._labels):
            dictionary[label] = len(self._label_sets[i])
        return dictionary

    def get_mask(self, label_name, object_number=None, binary=True):
        """Returns a boolean mask

        :param label_name: the label to mask eg. 'tomato'
        :param object_number: optional object number
        :param binary: weather to return a binary bitmask or retain the object
            values. Only works when object_number is None
        :return: An array the same size as the image with 1 representing the
            object and 0 elsewhere. If no object is specified, all objects
            with matching label will be masked as 1
        """
        try:
            label_index = self._labels.index(label_name)
        except ValueError:
            return None

        if object_number is None:
            data = self._label_single if binary else self._label_objects
            return data[label_index]
        else:
            try:
                #print("label index: {}, object num: {}".format(label_index, object_number))
                #print("objects in label index {}:{}".format(label_index, self._label_sets[label_index].keys()))
                return self._label_sets[label_index][object_number]
            except KeyError:
                return None

    def get_dense_mask(self, label_array, background_first=True):
        """Return a densely labeled image consisting of classes in label_name,
            plus an additional class indicating the background.

        :param label_array: A list of label strings
        :param background_first: Whether the background layer should be mask value 0
        :return:
        """
        shape = (len(label_array) + 1,) + self._image_shape
        masks = numpy.zeros(shape, dtype=numpy.uint32)

        # Build a priority label by removing the intersection of future
        # labels with what we've already collected
        for i, label in enumerate(label_array):
            mask = self.get_mask(label)
            if mask is None: continue
            mask = mask.astype(numpy.uint8)
            known_masks = mask * numpy.logical_or.reduce(masks)
            masks[i+background_first] = numpy.logical_xor(mask, known_masks)
        background_index = 0 if background_first else -1
        masks[background_index] = numpy.bitwise_not(numpy.logical_or.reduce(masks))
        # Give each channel a unique label by multiplying its boolean
        # mask by an array indicating its index in the array
        masks = masks * numpy.arange(shape[0]).reshape(shape[0], 1, 1)
        return numpy.sum(masks, axis=0, dtype=numpy.uint8)

    def get_object_masks(self, class_name):
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

        image_shape = self._image_shape
        object_count_dict=self.get_classes()

        if class_name not in object_count_dict.keys():
            return []

        num_objects = object_count_dict[class_name]
        object_masks = np.zeros((num_objects, ) + image_shape, np.uint8)


        # Loop over all the objects
        for i in xrange(num_objects):

            mask = self.get_mask(class_name, i + 1)

            if mask is None:
                continue
            object_masks[i] = np.uint8(mask * 255)

        return object_masks

def read(annotation_path):
    """Parses annotation data from a .png file.

    :param annotation_path: The path to the annotation .png file
    :return: an AnnotationData object containing the loaded data
    """
    #print("Reading: {}".format(annotation_path))
    try:
        image_data = numpy.array(Image.open(annotation_path))
    except IOError:
        print("Cannot find annotation file")
        return None

    # Read metadata
    binary_png = open(annotation_path, "rb")
    binary_png.seek(33)
    data_length = struct.unpack(">L", binary_png.read(4))[0]
    binary_png.seek(41)
    metadata = binary_png.read(data_length)
    metadata = json.loads(metadata)

    return AnnotationData(image_data, metadata)
