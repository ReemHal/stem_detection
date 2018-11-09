#!/usr/bin/env python
import os
from skimage.segmentation import slic
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage import io
import cv2
import bisect
import math
import copy
from PIL import Image
import seaborn as sns

class ImgProc:

    def __init__(self, image, debugging=False, area_thresh=None, x_thresh=None, y_thresh=None,\
                    maximum_edge_point_distance = None, angle_thresh = None, slic_pars=None, \
                    border_dict=None, key=None):
        self.key=key
        self.image = image
        self.debugging = False
        if (area_thresh ==  None):
            self.area_thresh = {'stem':200, 'tomato':100, 'leaf':100} # used to prune out prediction segments that are too small.
        if (x_thresh ==  None):
            self.x_thresh = {'stem':100, 'tomato':10, 'leaf':0} # defines the smallest acceptable length along the x-axis in the ground truth masks
        if (y_thresh ==  None):
            self.y_thresh = {'stem':100, 'tomato':10, 'leaf':0} # defines the smallest acceptable length along the y-axis in the ground truth masks
        if (slic_pars==None):
            self.slic_pars={'tomato':
                                {'compactness_val': 10.0,
                                 'numSegments_val':500,
                                 'sigma_val':5,
                                 'slic_zero_mode':False
                                },
                            'stem':
                                {'compactness_val': 10.0,
                                 'numSegments_val':100,
                                 'sigma_val':5,
                                 'slic_zero_mode':True
                                },
                            'leaf':
                                {'compactness_val': 10.0,
                                 'numSegments_val':100,
                                 'sigma_val':5,
                                 'slic_zero_mode':True
                                }
                            }
        if (border_dict==None):
            self.border_dict={'tomato':'tomato border',\
                              'leaf':'leaf border',\
                              'stem':'stem border'}
        if maximum_edge_point_distance == None:
            self.maximum_edge_point_distance = 20
        if angle_thresh == None:
            self.angle_thresh = 15

    def showme(self, img, img_title=''):
        fig=plt.figure()
        fig=fig.add_subplot(111)
        fig.imshow(img, cmap='plasma')
        plt.title(img_title)
        plt.show()

    def showComponents(self, mask):
        """ Displays each connected component indepentdently in a given class mask (superpixels)"""

        from skimage import measure

        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        labels = measure.label(thresh, neighbors=8, background=0)
        for label in range(0,len(labels)):
            img = np.zeros(mask.shape)
            # if this is the background label, ignore it
            if label == 0:
                continue
            img[labels==label]=255
            numPixels = cv2.countNonZero(img)

        	# if the number of pixels in the component is sufficiently
        	# large, then add it to our mask of "large blobs"
            if numPixels > 500:
                showme(img, 'Contour '+str(label))

    def get_SLIC_mask(self, class_mask, compactness_val=10.0, numSegments_val=100, sigma_val=5, slic_zero_mode=True):
        """Generate the boundaries of superpixels for a given class mask in the image with superpixels
        """

        # get the superpixels mask with the number of segments as set for the current label
        slic_superpixels = slic(img_as_float(self.image), compactness=compactness_val, \
                                n_segments=numSegments_val, sigma=sigma_val, \
                                convert2lab=True, slic_zero=slic_zero_mode) #n_segments= numSegments, sigma= sigmaVal,
        all_slic_contours = self._find_SLIC_boundaries(slic_superpixels).astype(np.uint8)
        slic_superpixels[class_mask==0]=0
        slic_superpixels[slic_superpixels>0]=255
        #kernel = np.ones((5,5),np.uint8)
        #slic_superpixels = cv2.morphologyEx(slic_superpixels.astype(np.uint8), cv2.MORPH_CLOSE, kernel)#, iterations=3)
        slic_contours = self._find_SLIC_boundaries(slic_superpixels).astype(np.uint8)

        return all_slic_contours,slic_contours, slic_superpixels

    def get_class_stats(self, curr_class, class_superpixels, kernel_size=3):
        """Returns a dictionary of stats for predicted curr_class segmentations."""

        # Create a label for each component in the image
        num_obj, output, bbox, centroids = \
            cv2.connectedComponentsWithStats(class_superpixels.astype(np.uint8), connectivity=8) #added dtype uint8

        # Only keep blobs that are above a certain area threshold. Zero out all
        # blobs that are below this threshold. bbox[i,-1] returns the area of the bounding box
        good_idx = [i for i in xrange(num_obj) if 50000 > bbox[i, -1] > self.area_thresh[curr_class]] # here changes max from 10000

        # if no blobs / objects were found
        if len(good_idx) == 0:
            return None, None
        good_idx = np.hstack(good_idx)

        # Zero out all detections on the image
        for i in set(np.arange(num_obj)).symmetric_difference(good_idx):
            output[output == i] = 0

        stats = {}
        stats['count'] = len(good_idx)
        stats['bbox'] = bbox[good_idx] #(tl_x, tl_y, width, height, area)
        stats['centroid'] = centroids[good_idx].astype(np.int32)

        # keep track of how to map labels to stats indices
        idx_map={}
        count=0
        for i in range(0,len(good_idx)):
            idx_map[good_idx[i]]=i+1

        for i in range(0,len(good_idx)):
            output[output==good_idx[i]]=idx_map[good_idx[i]]

        return stats, output

    def consolidate_instances_all_way(self, stats, segmented_instances):
        """ Given a set of segmented instances in a segmented_instances mask,
            cluster them by segment slope and centroid slope similarity.
            Used to find full stems given their disjoint parts
        """

        img = np.zeros(segmented_instances.shape).astype(np.uint8)

        #get all pixel labels in the segmented_instances mask
        segment_numbers = np.unique(segmented_instances)

        # remove the background label
        segment_numbers=segment_numbers[segment_numbers!=0]

        end_points = np.empty((len(segment_numbers),),dtype=np.object_)
        end_points.fill([])

        for curr_segment in segment_numbers:
            idx=[]
            i=curr_segment-1
            if curr_segment!=0:
                #Show all segments of curr_segment. Only useful to view results
                img[segmented_instances== curr_segment]= 255
                #get indeces of the segments for curr_segment
                idx = np.argwhere(segmented_instances == curr_segment)
                if len(idx>0):
                    end_points[i]= self._get_end_points(segmented_instances, i, \
                                                       stats, idx)
                    # add point markers and lines connecting each end point to centroid.
                    # useful only to view results
                    """for pt_num, pt in enumerate(end_points[i]):
                        cv2.circle(img, (pt[0],pt[1]), 3, 100, -1)
                        cv2.line(img,(pt[0],pt[1]),\
                                     (stats['centroid'][i,0], stats['centroid'][i,1]),150,2)
                    cv2.circle(img, (stats['centroid'][i,0], stats['centroid'][i,1]), 3, 200, -1)"""
        #self.showme(img, 'line '+str(i))

        # cluster segments into stem instances
        cluster_mask, clustered_instances = self._cluster_segments_all_way(segmented_instances,\
                                                    segment_numbers, end_points, \
                                                    stats)

        #put all instances in one layer
        if len(cluster_mask)>0:
            single_layer_cluster_mask=np.zeros(cluster_mask[0].shape)
            for i in xrange(len(cluster_mask)):
                single_layer_cluster_mask[cluster_mask[i]>0]= i+1

            # self.showObjects(clustered_instances);
        return single_layer_cluster_mask, clustered_instances

    def get_watershed(self, class_mask, dist_thresh=0.6, kernel_size=3, area_thresh=500):
        """ Returns the contour and superpixels in self.image for a given class mask"""

        watershed_contours = self._get_watershed_boundaries(class_mask, dist_thresh)
        watershed_areas = self._get_watershed_areas(watershed_contours, class_mask, \
                                                   kernel_size, area_thresh)

        return watershed_contours, watershed_areas

    def consolidate_instances(self, stats, segmented_instances, idx_map):
        """ Currently not used.
            Uses stem segment slopes to cluster segments into indipendent instances.
            Replaced by consolidate_instances_all_way"""

        img = np.zeros(segmented_instances.shape).astype(np.uint8)

        labels = np.unique(segmented_instances)
        labels=labels[labels!=0]
        reverse_idx_map = np.zeros(len(idx_map)).astype(np.int)
        for l in labels:
            reverse_idx_map[idx_map[l]]=np.int(l)

        #calculate slope of line between centroids.
        # TO DO: make this more efficient.
        centroid_slopes = self._calc_centroid_slopes(segmented_instances, labels, stats, idx_map)
        seg_slopes = np.zeros(len(labels))
        #for each instance i
        for i in range(0, len(labels)):
            idx=[]
            curr_label = reverse_idx_map[i]
            if curr_label!=0:
                #Show all segments of curr_label
                img[segmented_instances== curr_label]= 255
                #calculate slope m of instance i
                idx = np.argwhere(segmented_instances == curr_label)
                if len(idx>0):
                    max_y= max(idx[:,0])
                    min_y= min(idx[:,0])
                    x_for_max = idx[idx[:,0]==max_y, 1][0]
                    x_for_min = idx[idx[:,0]==min_y, 1][0]
                    if x_for_max < x_for_min:
                        x1= x_for_max
                        y1= max_y
                        x2= x_for_min
                        y2= min_y
                    else:
                        x1= x_for_min
                        y1= min_y
                        x2= x_for_max
                        y2= max_y
                    m = self._slope(x1,y1,x2,y2)
                    seg_slopes[i]=m
                    cv2.line(img,(x1, y1),(x2, y2),(0,100,0),4)
                    cv2.circle(img, (stats['centroid'][i,0], stats['centroid'][i,1]), 3, (200, 0, 0), -1)
                #self.showme(img, 'line '+str(i))

        # cluster segments
        clusters, clustered_instances = self._cluster_segments(segmented_instances, centroid_slopes, seg_slopes, reverse_idx_map)
        #find the closest centroid to a line with slope m that starts at the instances centroid
        # self.showObjects(clustered_instances);
        return clusters, clustered_instances

    def get_main_class_objects(self, curr_class, annotated):

        yt_class_objects = annotated.get_object_masks(curr_class)
        if yt_class_objects==[]:
            return []

        # Remove small objects marked as stems
        max_i = yt_class_objects.shape[0]
        i=0
        while i< max_i:
            class_obj_pixels= np.nonzero(yt_class_objects[i])
            if len(class_obj_pixels[1])>0:
                min_x =min(class_obj_pixels[1])
                max_x =max(class_obj_pixels[1])
                min_y =min(class_obj_pixels[0])
                max_y =max(class_obj_pixels[0])

                if max_x-min_x<self.x_thresh[curr_class] and max_y-min_y<self.y_thresh[curr_class]:
                    mask = np.ones(yt_class_objects.shape[0], dtype=bool)
                    mask[i]=False
                    yt_class_objects=yt_class_objects[mask]
                    max_i=max_i-1

                    """if i< max_i and curr_class=='stem':
                        self.showme(yt_class_objects[i], 'after')"""
                else:
                    """if curr_class=='stem':
                        self.showme(yt_class_objects[i], 'nothing changed')
                    """#showme(yt_class_objects[i].astype(np.uint8),'instance'+str(i)+'area:'+str(np.count_nonzero(yt_class_objects[i])))
                    i=i+1
            else: # remove layers with no objects
                """print "nothing to see here"
                if curr_class=='stem':
                    self.showme(yt_class_objects[i], 'nothing to see')"""
                mask = np.ones(yt_class_objects.shape[0], dtype=bool)
                mask[i]=False
                yt_class_objects=yt_class_objects[mask]
                max_i=max_i-1

        return   yt_class_objects

    def generate_detections_composite(self, curr_class, instances, stats, key, mark_centroids=False):
        """Given the current class and its clustered instances (for stems) or
           segmented_instances (for all other categories), it retruns an image showing:
            1. the centroids of detected isntances in red.
            2. stem instances where for each stem all its segments have the same color.
        """

        if curr_class == 'stem':
            composite = np.tile(instances[:, :, np.newaxis], (1, 1, 3))
            for i in range(0,composite.shape[0]):
                for j in range(0,composite.shape[1]):
                    composite[i,j]= composite[i,j]*[255, 150, 50]
        else:
            composite = np.tile(instances[:, :, np.newaxis], (1, 1, 3))
            composite[composite > 0] = 255

        composite = composite.astype(np.uint8)
        composite = np.uint8(0.6*composite) + np.uint8(0.5*self.image)

        if mark_centroids==True:
            for (i, c) in enumerate(stats['centroid']):
                color=(255,0,0)
                cv2.circle(composite, (c[0], c[1]), 5, color, -1)

        return composite

    def add_groundtruth_centroids(self, curr_class, key, dataset_json, composite):
        # Draw the ground-truth centoids using a green circle, and predicted
        # centroids using a smaller red circle
        for c in dataset_json[curr_class][key]['centroid']:
            cv2.circle(composite, (c[0], c[1]), 3, (0, 255, 0), -1)

        return composite

    def get_instance_mask(self, segMode, dilateFlag, key, rgb_image, curr_class, class_labels,\
                          segementation_function, mark_centroids=False, use_crf=False):

        # Get the regular & post-processed segmentations from the network.
        # Here, we use the post-processed output for segmenting tomatoes
        output, crf_output = segementation_function(rgb_image)
        if use_crf==True:
            segmentation = crf_output
        else:
            segmentation= output

        """self.save_img(rgb_image[0], key, prefix='reg', save_dir='../output/sample-seg')
        self.save_img(output, key, prefix='seg', save_dir='../output/sample-seg')
        self.save_img(crf_output, key, prefix='crf', save_dir='../output/sample-seg')"""

        # We use the watershed or SLIC algorithm for further segmenting predictions
        # made by the network. These algorithms require labels for where we "think"
        # points of interest area, and areas of the image which are definitely
        # background.
        class_mask = np.where(segmentation == class_labels[curr_class], 1, 0)
        class_mask = class_mask.astype(np.uint8)
        border_class_mask= np.where(segmentation == class_labels[self.border_dict[curr_class]], 1, 0)
        border_class_mask = class_mask.astype(np.uint8)

        if (segMode == 'watershed'):
            watershed, watershed_superpixels = self.get_watershed(class_mask, self.area_thresh[curr_class])
            segmented_img = watershed
            segmented_superpixels = watershed_superpixels
        elif (segMode == 'slic'):
            all_slic_contours, slicSeg, slic_superpixels = self.get_SLIC_mask(class_mask, compactness_val=self.slic_pars[curr_class]['compactness_val'], \
                                                                       numSegments_val=self.slic_pars[curr_class]['numSegments_val'], \
                                                                       sigma_val=self.slic_pars[curr_class]['sigma_val'],\
                                                                       slic_zero_mode=self.slic_pars[curr_class]['sigma_val'])

            segmented_img = slicSeg
            segmented_superpixels= slic_superpixels
        else:
            segmented_superpixels=class_mask

        # For each segmented superpixel, pass it through
        # cv2.connectComponentsWithStats and perform some extra preprocessing
        # to find where each centroid is. We also threshold predicted contours
        # that are below a certain area size to avoid false detection /
        # multiple detections on a single object
        stats, segmented_instances = \
            self.get_class_stats(curr_class, segmented_superpixels)
        # slic generates superpixels that are thinner than the actual instance. dilation helps overcome this issue.
        kernel = np.ones((5,5),np.uint8)
        instance_mask=[]
        composite_img=[]
        if (len(np.unique(segmented_instances))>0) and (segmented_instances is not None):
            """plt.imshow(segmented_instances)
            plt.title('segs')
            plt.show()"""
            if dilateFlag==True:
                dilated_seg_instances= cv2.dilate(segmented_instances.astype(np.uint8), \
                                                    kernel, iterations=1)
                instance_mask= dilated_seg_instances
            else:
                instance_mask= segmented_instances
            if curr_class == 'stem':
                """tmp=self.image.copy()
                tmp[slicSeg>0]=200
                self.save_img(tmp, key, prefix='stem_only', save_dir='../stem_superpixel_boundaries')
                tmp=self.image.copy()
                tmp[all_slic_contours>0]=200
                self.save_img(tmp, key, prefix='all', save_dir='../stem_superpixel_boundaries')"""

                cluster_mask, clustered_instances = self.consolidate_instances_all_way(stats, \
                                                        instance_mask)
                instance_mask = cluster_mask
                #self.showme(instance_mask)
            if (stats is None) or (stats == []):
                print 'No '+curr_class+' instances detected in image.'
                return [], [], []

            # ------------------- PLOT DETECTIONS -----------------------------
            # First convert the tomato mask into an RGB image by tiling it across
            # the channel dimension, then linearly blend with the original RGB image.
            composite_img= self.generate_detections_composite(curr_class, \
                                                                      instance_mask, stats, key, mark_centroids)

            #print "data:", stats['centroid'],dataset_json[curr_class][key]['bbox'],dataset_json[curr_class][key]['centroid']
        return instance_mask, composite_img, stats

    def save_img(self, input_image, key, prefix=None, save_dir='samples'):
        """Save an RGB image"""
        print "saving to...", input_image.shape, len(input_image.shape)

        fileName=key.rsplit('/', 1)[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        im=input_image.copy()
        if len(input_image.shape)==2:
            im = self.map_pixels(im)

        elif len(input_image.shape)==3:
            im= im.transpose(1, 2, 0).astype(np.uint8)
            im = im[...,[2,1,0]]
        if prefix is None:
            print os.path.join(save_dir, str(fileName)+'.jpeg')
            cv2.imwrite(os.path.join(save_dir, str(fileName)+'.jpeg'), im)
        else:
            print os.path.join(save_dir, str(fileName)+'-'+str(prefix) +'.jpeg')
            cv2.imwrite(os.path.join(save_dir, str(fileName)+'-'+str(prefix) +'.jpeg'), im)

    def map_pixels(self, mask, cmap=None):
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

    def _calc_centroid_slopes(self, main_img, labels, stats, idx_map):
        num_labels = len(labels)
        centroid_slopes = np.zeros((num_labels, num_labels))
        for i in range(0,num_labels):
            img = main_img
            for j in range(i+1, num_labels):
                c1 = stats['centroid'][i]
                c2 = stats['centroid'][j]
                if c1[0]<c2[0]:
                    x1 = c1[0]
                    x2 = c2[0]
                    y1 = c1[1]
                    y2 = c2[1]
                else:
                    x1 = c2[0]
                    x2 = c1[0]
                    y1 = c2[1]
                    y2 = c1[1]
                centroid_slopes[i,j] = self._slope(x1,y1,x2,y2)
                centroid_slopes[j, i] = centroid_slopes[i,j]
                #cv2.line(img,(x1, y1),(x2, y2),(255,0,0),4)
                #self.showme(img, 'centroid slopes '+str(i) + str(j))
        return centroid_slopes

    def _get_watershed_boundaries(self, class_mask, dist_thresh=0.6):
        """Creates a contour mask of watershed superpixels given class mask from VGG network."""

        kernel = np.ones((5, 5), np.float32)

        # Use a distance transform to find the seed points for watershed
        tmp = class_mask
        tmp[tmp>0] = 1 # here
        dist = cv2.distanceTransform(tmp, cv2.DIST_L2, 5) # here .astype(np.uint8), cv2.DIST_L2, 5)
        dist = (dist / np.max(dist)) * 255.

        # Since there may be multiple peaks, we use dilation to find them
        dilate = cv2.dilate(dist, kernel, iterations=3)
        peaks = np.float32(np.where(dilate == dist, 1, 0))
        peaks = peaks * class_mask * 255

        sure_fg = np.where(peaks > 125, 255., 0.)
        sure_fg = cv2.dilate(sure_fg, kernel, iterations=2)
        sure_fg = np.uint8(sure_fg)

        sure_bg = cv2.dilate(class_mask, kernel, iterations=3) * 255
        unknown = sure_bg - sure_fg

        # Add one to all labels so that known background is not 0, but 1
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1

        markers[unknown == 255] = 0

        markers = cv2.watershed(self.image, markers)

        watershed_superpixels = np.zeros(class_mask.shape, dtype=np.uint8)
        watershed_superpixels[markers == -1] = 255

        return watershed_superpixels

    def _get_watershed_areas(self, class_contours, class_mask, kernel_size=3, area_thresh=500):
        """ Returns the areas within a given watershed contours"""

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        # Since the watershed draws contours, we need to invert the predictions to
        # get the 'inside' blob portion. We also slightly compress the blob portion
        # so we can get a more defining border.
        inverted_contours = 255 - class_contours

        inverted_contours = cv2.erode(inverted_contours, kernel, iterations=1)
        # remove areas that are not part of the class mask
        inverted_contours[class_mask==0]= 0 # here ?

        return inverted_contours

    def _find_SLIC_boundaries(self, segments):
        float_image= np.zeros(segments.shape, dtype=np.uint8)
        slic_boundaries = find_boundaries(segments,  mode='outer', connectivity=2)

        return slic_boundaries.astype(np.uint8)

    def _slope(self, x1,y1, x2,y2):
        if x1==x2:
            return None
        return float(y1-y2)/float(x1-x2)

    def _cluster_segments(self, segmented_instances, centroid_slopes, seg_slopes, reverse_idx_map, cluster_thresh=0.6):
        #self.showme(segmented_instances, 'main img')
        slope_diff = []
        visited={}
        used={}
        for i in range(0, centroid_slopes.shape[0]):
            visited[i]=False
            used[i]=False
            for j in range(i+1, centroid_slopes.shape[1]):
                diff = abs(seg_slopes[i]-centroid_slopes[i,j]) + abs(seg_slopes[j]-centroid_slopes[i,j])
                #direction = 1 if (seg_slopes[i]*centroid_slopes[i,j])> 0 :0
                if diff < cluster_thresh:
                    slope_diff.append((i,j,diff))

        slope_diff = sorted(slope_diff,key=lambda x:(x[2],-x[1]))

        # find best match by iteretively selecting the smallest difference
        tmp = np.full(reverse_idx_map.shape, None)
        for curr_tuple in slope_diff:
            i = curr_tuple[0]
            j = curr_tuple[1]
            if (visited=={} or visited[i]==False) and (used=={} or used[j]==False):
                tmp[i]=j
                visited[i]=True
                used[j] = True
        clusters = {}
        visited= dict.fromkeys(visited, False)
        cluster_num=1
        clusterImg = np.zeros(segmented_instances.shape).astype(np.uint8)
        for i in range(0, len(tmp)):
            clusters[cluster_num-1]= []
            seg = i
            newCluster = False
            while (seg != None) and (visited[seg]!=True):
                clusters[cluster_num-1].append([seg])
                clusterImg[(segmented_instances == reverse_idx_map[seg])] = cluster_num
                visited[seg]=True
                seg = tmp[seg]
                newCluster=True
            if newCluster == True:
                cluster_num += 1
                #self.showme(clusterImg*20, str(i))
        return clusters, clusterImg

    def _get_centeroids(self, pts, max_dist=None):
        """ Get the center points of all points in pts.
            All elements in pts fall on a single vertical or horizontal line.
        """

        if max_dist == None:
            max_dist=self.maximum_edge_point_distance
        dist=0
        center_pts=[]
        if len(pts)>0:
            pts = sorted(pts)
            i = 0
            j=1
            center_pts.append(pts[0])
            num_pts=1.0 # to avoid integer division
            num_clusters = 1
            while i<len(pts) and j<len(pts):
                if pts[j]-pts[i]<max_dist:
                    if len(center_pts)<num_clusters:
                        center_pts.append(0)
                    center_pts[num_clusters-1] = center_pts[num_clusters-1]+pts[j]
                    num_pts+=1.0
                    j=j+1
                    i+=1
                else:
                    if len(center_pts)<=num_clusters:
                        center_pts.append(0)
                    center_pts[num_clusters-1]= int(center_pts[num_clusters-1]/num_pts)
                    num_pts = 1
                    num_clusters += 1
                    i=j
                    j=i+1
            if len(center_pts)<=num_clusters:
                center_pts[num_clusters-1] = int(center_pts[num_clusters-1]/num_pts)
            else:
                print "error! center_pts!", len(center_pts), num_clusters, self.key
        return center_pts

    def _update_imgs_and_pt_list(self, points, edge_points, segs, index):
        """ Put all points of a specified coordinate in x_pts"""
        # index specifies whether to use the x or y coordinate in x_pts
        x_pts=[]
        for i in range(0, len(points)):
            pt=points[i]
            #edge_points[pt[0],pt[1]] = 255
            x_pts.append(pt[index])
            #segs[pt[0],pt[1]]=150

        return x_pts, segs, edge_points

    def _get_corner_centers(self, y_pts, x_pts,y_val, x_val,  max_dist = None):
        """ If a stem end falls on a corner, it will have some of the end's pixel
            on one side of the bounding box and others on another, perpendicular, side.
            This function replaces all those corner pixels with their centroid """

        if max_dist == None:
            max_dist=self.maximum_edge_point_distance

        min_dist= None
        center_x=None
        center_y=None
        # Calculate the centroid of the corner pixels
        for i, l_pt in enumerate(y_pts):
            for j, b_pt in enumerate(x_pts):
                dist = math.sqrt((y_val-l_pt)**2.0+ (x_val-b_pt)**2.0)
                if min_dist == None or min_dist > dist:
                    min_dist = dist
                    center_y = int(abs(y_val - l_pt)/2.0) + min(l_pt,y_val)
                    center_x = int(abs(b_pt- x_val)/2.0) + min(x_val, b_pt)
                    y_i =i
                    x_j = j

        if (min_dist<max_dist) and (min_dist != None):
            #remove points for which we have a centroid and all nearby points
            y_pts = self._delete_point_cluster(y_pts, y_i, max_dist)
            x_pts = self._delete_point_cluster(x_pts, x_j, max_dist)
        else:
            center_x=None
            center_y=None

        return center_x, center_y, y_pts, x_pts

    def _delete_point_cluster(self, pts, starting_pt, max_dist=None):
        """ delete all pixels that belong to the same cluster as starting_pt """

        if max_dist == None:
            max_dist=self.maximum_edge_point_distance
        first = starting_pt
        last = first
        # Find end of cluster
        k = last+1
        while last < len(pts)-1 and k< len(pts)-1 and abs(pts[k]-pts[last])<max_dist:
            k = k+1
            last+=1
        # Find beginning of cluster
        k=first-1
        while first> 0 and k> 0 and abs(pts[k]-pts[first])<max_dist:
            k=k-1
            first=first-1

        # Delete cluster pixels
        for i in range(first,last+1):
            del pts[i]

        return pts

    def _get_end_points(self, segmented_instances, i, stats, idx):
        """ get the points instersecting with the bounding box for segment identified by idx indeces"""

        end_points=[]

        # find all points intersecting the bbox
        #(tl_x, th_y, width, height, area)
        label_num=i+1
        leftmost_x = stats['bbox'][i][cv2.CC_STAT_LEFT]
        topmost_y = stats['bbox'][i][cv2.CC_STAT_TOP]
        width = stats['bbox'][i][cv2.CC_STAT_WIDTH]
        height = stats['bbox'][i][cv2.CC_STAT_HEIGHT]
        bottom_most_y = topmost_y + height-1
        right_most_x = leftmost_x + width-1

        segmented_instances_copy=segmented_instances.copy()
        edge_points = np.zeros(segmented_instances.shape).astype(np.uint8)
        segs = np.zeros(segmented_instances.shape).astype(np.uint8)
        segs[segmented_instances==label_num]=255
        cv2.rectangle(segmented_instances_copy,(leftmost_x, topmost_y), (right_most_x, bottom_most_y), 150, 2)

        #Get all points for the current stem segment
        label_points = np.argwhere(segmented_instances.copy()==label_num)

        # upper points from (tl_x,th_y) to (th_x, th_y) that instersect with the upper edge of the bouding box
        upper_points = [i for i in label_points if  i[0]==topmost_y and i[1]>=leftmost_x and i[1]<=right_most_x]
        x_pts, segs, edge_points = self._update_imgs_and_pt_list(upper_points, edge_points, segs, 1)
        center_upper_pts = sorted(self._get_centeroids(x_pts))

        # left side points from (tl_x, tl_y) to (tl_x, th_y) that instersect with the left edge of the bouding box
        left_points = [i for i in label_points if i[1]==leftmost_x and i[0]<=bottom_most_y and i[0]>=topmost_y]
        x_pts, segs, edge_points = self._update_imgs_and_pt_list(left_points, edge_points, segs, 0)
        center_left_pts = sorted(self._get_centeroids(x_pts))

        #right side points form (th_x, tl_y) to (th_x, th_y) that instersect with the right edge of the bouding box
        right_points =  [i for i in label_points if i[1]==right_most_x and i[0]<=bottom_most_y and i[0]>=topmost_y]
        x_pts, segs, edge_points = self._update_imgs_and_pt_list(right_points, edge_points, segs, 0)
        center_right_pts = sorted(self._get_centeroids(x_pts))

        #bottom points from (tl_x, tl_y) to (th_x,tl_y)
        bottom_points =  [i for i in label_points if i[1]>=leftmost_x and i[1]<=right_most_x and i[0]==bottom_most_y]
        x_pts, segs, edge_points = self._update_imgs_and_pt_list(bottom_points, edge_points, segs, 1)
        center_bottom_pts = sorted(self._get_centeroids(x_pts))

        # If there are corner edges, get the centroid of that
        center_x_lb, center_y_lb, center_left_pts, center_bottom_pts = self._get_corner_centers(center_left_pts, \
                                                                center_bottom_pts, bottom_most_y, leftmost_x)
        if (center_x_lb != None) and (center_y_lb != None):
            end_points.append([center_x_lb, center_y_lb])
        else:
            if len(center_left_pts)>0:
                for pt_idx in range(0, len(center_left_pts)):
                    end_points.append([leftmost_x, center_left_pts[pt_idx]])
            if len(center_bottom_pts)>0:
                for pt_idx in range(0, len(center_bottom_pts)):
                    end_points.append([center_bottom_pts[pt_idx], bottom_most_y])

        # If there are corner edges, get the centroid of that
        center_x_ur, center_y_ur, center_right_pts, center_upper_pts = self._get_corner_centers(center_right_pts, \
                                                                center_upper_pts, topmost_y, right_most_x)
        if (center_x_ur != None) and (center_y_ur != None):
            end_points.append([center_x_ur, center_y_ur])
        else:
            if len(center_right_pts)>0:
                for pt_idx in range(0, len(center_right_pts)):
                    end_points.append([right_most_x, center_right_pts[pt_idx]])
            if len(center_upper_pts)>0:
                for pt_idx in range(0, len(center_upper_pts)):
                    end_points.append([center_upper_pts[pt_idx], topmost_y])

        # If there are corner edges, get the centroid of that
        center_x_ul, center_y_ul, center_left_pts, center_upper_pts = self._get_corner_centers(center_left_pts, \
                                                                center_upper_pts, topmost_y, leftmost_x)
        if (center_x_ul != None) and (center_y_ul != None):
            end_points.append([center_x_ul, center_y_ul])
        else:
            if len(center_left_pts)>0:
                for pt_idx in range(0, len(center_left_pts)):
                    end_points.append([leftmost_x, center_left_pts[pt_idx]])
            if len(center_upper_pts)>0:
                for pt_idx in range(0, len(center_upper_pts)):
                    end_points.append([center_upper_pts[pt_idx], topmost_y])


        # If there are corner edges, get the centroid of that
        center_x_br, center_y_br, center_right_pts, center_bottom_pts = self._get_corner_centers(center_right_pts, \
                                                                center_bottom_pts, bottom_most_y, right_most_x)
        if (center_x_br != None) and (center_y_br != None):
            end_points.append([center_x_br, center_y_br])
        else:
            if len(center_right_pts)>0:
                for pt_idx in range(0, len(center_right_pts)):
                    end_points.append([right_most_x, center_right_pts[pt_idx]])
            if len(center_bottom_pts)>0:
                for pt_idx in range(0, len(center_bottom_pts)):
                    end_points.append([center_bottom_pts[pt_idx], bottom_most_y])

        #self.showme(segmented_instances_copy, 'bbox')

        return end_points

    def _dot(self, vA, vB):
        return float(vA[0]*vB[0]+vA[1]*vB[1])

    def _ang(self, lineA, lineB, rounding_error_tolerance=5e-6):
        #ang_degree=None
        # Get nicer vector form
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Get dot prod
        dot_prod = self._dot(vA, vB)
        # Get magnitudes
        magA = math.sqrt(self._dot(vA, vA))
        magB = math.sqrt(self._dot(vB, vB))
        if (magB*magA)==0:
            return 999
        # Get cosine value
        cos_ = dot_prod/(magA*magB)
        if (math.fabs(cos_)>1.0) and (math.fabs(cos_)-1.0<rounding_error_tolerance):
            cos_ = math.modf(cos_)[1]
        if (math.fabs(cos_)<=1.0):
            # Get angle in radians and then convert to degrees
            angle = math.acos(cos_)
            # Basically doing angle <- angle mod 360
            ang_deg = math.degrees(angle)%360
        else:
            print "incorrect vactor values:", lineA , 'and', lineB, "dot:", dot_prod, 'magA',magA,'magB',magB
            print "cos: ", dot_prod/magB/magA, 'acos factor:', dot_prod/magA/magB, 'or:', dot_prod/(magB*magA)
            raise ValueError('Invalid arguments. The cosine value is not between -1 and 1!')
        return ang_deg

    def _get_best_fit(self, segmented_instances, num_labels,\
                      stats, end_points, i, j, k, pos_angle=True):
        """ given a centroid line ij, find the segment section in j that has the
            smallest angle from the centroid line """
        min_angle = None
        seg_section = None
        min_seg_dist = None

        img = np.zeros(segmented_instances.shape)
        img[segmented_instances== (i+1)]= 100
        #self.showme(img, str(i))
        img[segmented_instances== (j+1)]= 100
        #self.showme(img, str(i)+'and? '+str(j))
        cv2.circle(img, (end_points[i][k][0], end_points[i][k][1]), 3, 70, -1)
        seg_section=None
        mid_point_i=[0,0]
        for l in range(0, len(end_points[j])): #Iterate over all endpoints of segment j
            #cv2.line(img,(end_points[i][k][0], end_points[i][k][1]),(stats['centroid'][i,0], stats['centroid'][i,1]),100,4)
            seg_dist = math.sqrt((end_points[j][l][0]-end_points[i][k][0])**2.0 +
                                 (end_points[j][l][1]-end_points[i][k][1])**2.0 )
            #cv2.line(img,(end_points[j][l][0], end_points[j][l][1]),(stats['centroid'][i,0], stats['centroid'][i,1]),255,4)
            # Stem segments with side branches usually miss a good connection because the centroid is off
            # the end point's axis. To account for this we replace the centroind with a point that is closer to the end point.
            mid_point_i[0] = int(end_points[i][k][0]+ (stats['centroid'][i][0]-end_points[i][k][0])/4.0)
            mid_point_i[1] = int(end_points[i][k][1]+ (stats['centroid'][i][1]-end_points[i][k][1])/4.0)

            angle = self._ang([stats['centroid'][j],end_points[j][l]], \
                              [stats['centroid'][j], mid_point_i] )
            if angle==999: #check for a divide by zero error
                cv2.line(img,(stats['centroid'][j][0], stats['centroid'][j][1]),
                             (end_points[j][l][0], end_points[j][l][1]), 150, 4 )
                cv2.line(img,(stats['centroid'][j][0], stats['centroid'][j][1]),
                             (stats['centroid'][i][0], stats['centroid'][i][1]), 150, 4 )
                cv2.line(img,(end_points[j][l][0], end_points[j][l][1]),\
                                (end_points[i][k][0], end_points[i][k][1]),255,4)
                self.showme(img, str(i)+' '+str(j)+' '+str(angle))

            if (pos_angle and angle<=self.angle_thresh) or ( not(pos_angle) and angle>=360-self.angle_thresh):
                if min_seg_dist is None or seg_dist < min_seg_dist:
                    min_seg_dist = seg_dist
                    min_angle = angle
                    seg_section = l
            elif seg_dist<50:
                cv2.line(img,(stats['centroid'][j][0], stats['centroid'][j][1]),
                             (end_points[j][l][0], end_points[j][l][1]), 150, 4 )
                cv2.line(img,(stats['centroid'][j][0], stats['centroid'][j][1]),
                             (stats['centroid'][i][0], stats['centroid'][i][1]), 150, 4 )
                cv2.line(img,(end_points[j][l][0], end_points[j][l][1]),\
                                (end_points[i][k][0], end_points[i][k][1]),255,4)
                #self.showme(img, str(i)+' '+str(j)+' '+str(angle))

        return min_angle, seg_section, min_seg_dist

    def _cluster_segments_all_way(self, segmented_instances, labels, \
                                     end_points, stats, cluster_thresh=0.5):
        """ Given a set of segments and their end points,
            iteratively clusters segmetns that have the smallest
            difference between a segment end point and the line
            connecting the centroids of consecutive segments with
            preference given to the segments whose end points are closest
        """

        #self.showme(segmented_instances, 'main img')
        segment_association_list = []
        max_num_end_points= 0

        # for each stem segment
        for i in range(0, len(labels)):
            # each end point in the current segment i
            if max_num_end_points < len(end_points[i]):
                max_num_end_points = len(end_points[i])
            for k in range(0, len(end_points[i])):
                angle_list=[]
                #  find the segment that is most likely connected to segment i at end point[i][k]
                for j in range(0, len(labels)):
                    # make sure we are not trying to connect the segment to itself
                    if i!= j:
                        # angle calculates the angle between the line stats['centroid'][i]-end_points[i][k]
                        # and stats['centroid'][i]-stats['centroid'][j]

                        angle = self._ang([stats['centroid'][i],end_points[i][k]], \
                                          [stats['centroid'][i], stats['centroid'][j]] )
                        # if the angle value is within the acceptable range of +/- angle_thresh
                        if angle<=self.angle_thresh or angle>=360-self.angle_thresh:
                            other_angle, other_seg_section, end_point_dist  = self._get_best_fit(segmented_instances, \
                                                                                             len(labels), \
                                                                                            stats, end_points,\
                                                                                            i, j, k, pos_angle=angle<=self.angle_thresh)
                            # if the best fit segment also has a small angle between its
                            # end point-centroid line and centroid-centroid line,
                            # add it to segments connected to segment i
                            if other_angle!=None and other_angle<=self.angle_thresh:
                                angle_list.append((j, other_seg_section, other_angle, end_point_dist, angle))
                #Sort the list of stem segments connected to i by end_point_dist
                angle_list = sorted(angle_list, key=lambda x:x[3])
                #Sorting by the Euclidian distance of the end_point_dist and the other_angle does not change end result
                #angle_list = sorted(angle_list, key=lambda x:(math.sqrt(x[3]**2.0+x[2]**2.0)))
                # the angle value reflects how far segment k is from the straight line
                # going through the centroids
                if len(angle_list)>0:
                    # (i, j, k, l, angle between i and centroid line, angle between j and centroid line, distance between closest end points k in seg i and l in seg j)
                    segment_association_list.append((i,angle_list[0][0],k, angle_list[0][1], angle_list[0][4], angle_list[0][2], angle_list[0][3]))


        # sort slope differences in an increasing order
        segment_association_list = sorted(segment_association_list,key=lambda x:(x[6]))

        # find best match by iteretively selecting the smallest difference
        # and adding it to the ith cluster
        cluster_list = []
        cluster = np.full(len(labels),None)
        colored_clusterImg = np.zeros(segmented_instances.shape).astype(np.uint8)
        #clusterImg = np.zeros(segmented_instances.shape).astype(np.uint8)

        # initialize cluster list to single clusters contianing only each individual segment
        for i in range(0, len(labels)):
            cluster[i]=i
            cluster_list.append([i])
            #self.showme(clusterImg, str(i))

        visited=np.full((len(labels),max_num_end_points), False)

        #cluster=np.frompyfunc(list,1,1)(cluster) # allows us to append to only the specified list end_points[i]
        new_cluster_num=0
        color_offset=len(labels)

        # for each pair of segments in our list of best fit segments
        for curr_tuple in segment_association_list:
            img = np.zeros(segmented_instances.shape)
            i = curr_tuple[0] # index of first segment
            j = curr_tuple[1] # index of second segment in the tuple
            i_section = curr_tuple[2] #end point number in segment i
            j_section = curr_tuple[3] #end point number in segment j
            angle = curr_tuple[4]
            other_angle = curr_tuple[5]
            end_point_dist = curr_tuple[6] #distance between the connecting end points of segments i and j
            img[segmented_instances== i]= 255
            img[segmented_instances== j]= 255
            if (visited[i][i_section]==False)and(visited[j][j_section]==False):
                #cv2.line(clusterImg,(end_points[i][i_section][0],end_points[i][i_section][1]),\
                #             (end_points[j][j_section][0], end_points[j][j_section][1]),150,2)
                #self.showme(clusterImg, str(i))
                visited[i][i_section]=True
                visited[j][j_section]=True
                cluster_num = cluster[i]
                if cluster[i]!=cluster[j]:
                        other_cluster_num = cluster[j]
                        cluster_list[cluster_num] = list(set(cluster_list[cluster_num]+\
                                                             copy.deepcopy(cluster_list[other_cluster_num])))
                        # update cluster numbers for all segments moved into new cluster
                        for seg in cluster_list[other_cluster_num]:
                            cluster[seg]=cluster_num
                        # update cluster numbers for clusters larger than cluster to be removed
                        for idx in range(0, len(cluster)):
                            if (cluster[idx]>other_cluster_num):
                                cluster[idx]= cluster[idx]-1
                        del cluster_list[other_cluster_num]


        #show clustered segments
        color = 0
        cluster_num = 0
        cluster_mask=[]

        for c in cluster_list:
            color = color+0.1
            cluster_mask.append(np.zeros(segmented_instances.shape).astype(np.uint8))

            for i in c:
                cluster_mask[cluster_num][(segmented_instances == labels[i])]=1
                colored_clusterImg[(segmented_instances == labels[i])]= int(color*255)
            """if self.key in ['../data/images/image1672', '../data/images/image1289']:
                self.showme(colored_clusterImg)"""
            cluster_num +=1

        return cluster_mask, colored_clusterImg
