# semantic-segmentation

This project contains the code used for performing semantic segmentation and tomato detection for the University of Guelph's GIGAS project.

# Results:
* [Semantic Segmentation Results](./docs/semantic-seg-results.pdf)
* [Tomato Detection Results](./docs/detection-results.pdf)

# Required Software:
* Linux
* cv2 (v3.2 used here, see: https://www.youtube.com/watch?v=jS7To92cnds)
* Seaborn ```pip install seaborn```
* cPickle ```pip install cPickle```
* Theano (see: http://deeplearning.net/software/theano/install_ubuntu.html)
* Lasagne (see: http://lasagne.readthedocs.io/en/latest/user/installation.html)

Note: The CRFRNN Layer is from [https://github.com/HapeMask/crfrnn_layer](here) and created by user "HapeMask".

# Getting Started

Clone the repository:
```git clone https://github.com/mveres01/semantic-segmentation```

* For using a pre-trained segmentation network, download the weight file [here](https://owncloud.guelphrobolab.ca/index.php/s/qhACRRBe6C47Xa5), and place it in the ./data folder
* For training your own segmentation network, download the the pre-trained ImageNet weights [here](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl), and place it in the ./data folder



## Step 1: Download the images 
Download images from the robotics lab / GIGAS server. This will create a subdirectory in the root called 'data' which will hold the RGB and annotated images. 

```
cd lib
python download_images.py
```

## Step 2: Create the dataset
Once the images have been saved locally, we need to create the dataset we'll use for training the segmentation network. Preprocessing the data in this way allows us to quickly index data while training, and avoiding costly train-time I/O.

```
cd lib
python create_dataset.py
```

Here, we save several different files:
* _./data/loc-tomato-labels.json_: json file containing the bounding box and centroid for all tomatoes in each image
* _./data/segmentation-label-dict.json_: json file mapping annotation labels to a specific class, e.g. {0=stem, 1=tomato ... 7=background}
* _./data/seg-tomato-images.npz_: a file containing all RGB images we'll use for segmentation / detection
* _./data/seg-tomato-labels.npz_: a file containing a dense mask corresponding to each RGB image, and used for semantic segmentation.
* _./data/composites_: a directory of images showing the RGB + associated dense mask we created while preparing the dataset

The main data/folder should now have a structure similar to this:
```
-./data/
+ composites/
+ images/
+ loc-tomato-labels.json
+ segmentation-label-dict.json
+ seg-tomato-images.npz
+ seg-tomato-labels.npz
```

# Semantic segmentation

To perform semantic segmentation, a deep fully-convolutional neural network architecture known as a [U-Net](https://arxiv.org/abs/1505.04597) was chosen. Due to the relatively small number of labeled images, it uses pre-trained VGG16 weights for the encoder portion. 

Training is done on images of sizes 700x700, and adds dataset augmentation by randomly flipping an image left <--> right with a probability of p=0.5. As there are significant differences in the frequency of each class (e.g. there is significantly more "background" pixels in each image then "tomato" pixels), the loss function is weighted proportional to the frequency of each class. This helps ensure the network learns to predict a class for each pixel with even probability. 

A _conditional random field_ [CRFasRNN](https://github.com/HapeMask/crfrnn_layer) was employed to clean up the output image, and the user is able to choose whether they want the raw output, or cleaned output. See Step 3 below for an example of the difference.

## Step 3: Testing the segmentation network

A pre-trained segmentation model (trained-weights.npz) has been provided with this code. This file is around 150 MB in size, and should be placed in the ./data/ subdirectory. To test that the network works properly, try running the test file, which should save the networks output (raw segmenation and after being processed using a CRF) to the folder './output/sample-seg/. This file can also be used as a reference to see the basic components that are needed to obtain a segmented image using deep learning.

```
cd src
python test_segmentation.py
```

Using the provided image in the src/ folder, you should expect to get an output similar to the below images, where the different colours denote the different object classes. The left image represents the raw prediction from the network, while the right image is post-processed through a CRF. Notice how the raw output tends to over-predict object classes, while the crf-processed output tends to under-predict classes.

<p align="center">
  <img src="./docs/raw.jpeg" width="400" alt="Raw Output"/>
  <img src="./docs/crf.jpeg" width="400" alt="CRF Output"/>
</p>

## (Optional) Training the segmentation network

Once the dataset has been constructed, a network that performs semantic segmentation can be trained by running the main script:

```
cd src
python train-segmentation.py
```

On a GTX TITAN X, one epoch through the dataset should take between 5-6 minutes, and training should require between 10-20 epochs total. 

# Testing the detection algorithms

Two detection algorithms are implemented: One based on [Hough Circle](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html) detection, and another based on the [Watershed](http://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html) algorithm. Both algorithms use the output of the segmentation network, using the "tomato" class mask. Adding the artificial "tomato border" class above allows these detection algorithms to more easily seperate and detect the tomatoes within an image. 

For example, the input image and predicted tomato mask (i.e. output of the segmentation network) can be seen below:

<p align="center">
  <img src="./docs/detection-input-image.jpg" width="400"/>
  <img src="./docs/segmentation-tomato-mask.jpg" width="400"/>
</p>

To test the detection algorithms on the sample image, run the following: 
```
cd src
python detect-hough.py
python detect-watershed.py
```
The left image is the output from a Hough Circle detection, where detected circles are in white, and their predicted centroids in yellow. The right image is the output from the Watershed segmentation, and the segmented components are overalyed on the original image using a faint white colour. 

<p align="center">
  <img src="./docs/hough-image1075.jpg" width="400"/>
  <img src="./docs/watershed-image1075.jpg" width="400"/>
</p>

## Hough Circles

Hough Circles is a Computer Vision algorithm that (as the name implies) looks for circular objects in an image. It is not, however, a robust algorithm, and is susceptible to changes in the environment such as lighting. Here, we try to limit false-positive predictions by forcing it to look for circles only where tomatoes are predicted to be in the image. 

## Watershed

As the segmentation results for tomatoes are fairly strong, a different approach attempts to simply use these predictions as the output; the Watershed algorithm attempts to seperate joined tomatoes, and the resulting contours are labeled as a tomato. The below images represent "detected" tomatoes in different colours (left) and their predicted boundaries (right).

<p align="center">
  <img src="./docs/watershed-markers.jpg" width="400"/>
  <img src="./docs/watershed-output.jpg" width="400"/>
</p>


## Discussion: Segmentation failure modes and trends

There are several trends in the dataset to be aware of, that influence segmentation and detection accuracy: 

### Duplicate images and artificial light

Several of the collected images in the dataset were found to contain duplicated spatial  features. Further, these images contained features of bright / different coloured light  that are not expected to be encountered in real world situations. Most of these images  were found to appear below the suffix of 978 and were removed from the datasets.

<p align="center">
  <img src="./docs/duplicate_red.jpg" width="400"/>
  <img src="./docs/duplicate_white.jpg" width="400"/>
</p>

### Inconsistent spatial dimensions

The majority of collected images have a defined shape of 860x720 pixels. However, there were several images that did not follow this pattern, and had sightly  smaller dimensions. While segmentation algorithms (such as from neural networks)  are able to process images of varying sizes, it was chosen to remove these from the  dataset in order to maintain a sense of consistency. Future experimentation may  choose to add these back in.

### Label quality

Images have varying levels of the quality of ground-truth segmentations. Some images have grainy class-segmentation boundaries which may be a result of the annotation tool and handling light illumination. Other images have more clear and smooth class boundaries, and are more suited for training semantic segmentation algorithms as the predictions are allowed to be more confident.

<p align="center">
  <img src="./docs/labeled_grainy.jpg" width="400"/>
  <img src="./docs/labeled_smooth.jpg" width="400"/>
</p>

There are also many instances where objects in an image have been incorrectly labeled as foreground (when they should have been background) and vice-versa. This presents a significant issue in properly quantifying all algorithms performance.

# Future directions

An initial object detector has been proposed in this work, but the project may benefit from a more tailored approach, especially when more then one class of objects are required to be found within an image. Google has recently released a Tensorflow object detection API that implements many of the current state of the art approaches found (here)[https://techcrunch.com/2017/06/16/object-detection-api/]. Recommendations for object detectors (in terms of complexity) are:

* Faster Region(proposal)-Convolutional Neural Networks (Faster RCNN)[https://github.com/rbgirshick/py-faster-rcnn]. This is a common object detector created around 2015. There are several implementations in different programming languages available online.
* (Mask R-CNN)[https://arxiv.org/abs/1703.06870]: Current State of the art object detection model. This is an extension of Faster R-CNN.

## Relational Reasoning

Google DeepMind released a paper in 2017 about learning relations between objects in an image (here)[https://deepmind.com/blog/neural-approach-relational-reasoning/]. This is an interesting approach, but I don't think that this will fit with the current GIGAS project. One of the biggest factors is time complexity: Their system only works with few objects (e.g. 6 per image), while a typical image of in a Greenhouse could contain between 50-100 objects (split between tomatoes, stems, leafs, etc.). Further, the way they train their system (using comparisons between each pixel in an image) requires many GPUs and days of training time. 

A more straight forward approach for GIGAS to learning object relations is likely to use some form of Graph Theory. 



