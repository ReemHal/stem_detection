import os
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

image_width = 860

def validate_centroids(yp_centroids, yt_bbox, yt_centroids):
    """Calculates how many centroids are predicted per ground-truth / labeled
    contour, and the distance from predicted to labeled centroids.
    """

    centroid_idx = np.ones((len(yp_centroids),))*-1
    bbox_hits = np.zeros((len(yt_bbox),))

    # for each predicted centroid, check whether we've hit a labeled object by seeing if
    # the prediction is within a given bounding box
    for i, centroid in enumerate(yp_centroids):

        # go over all ground-truth boxes until you find a box where the centroid falls within it
        for j, bbox in enumerate(yt_bbox):

            # Check if the current predicted centroid is within the perimeter of bbox.
            # bbox starts from y value (column) bbox[0] with width bbox[2]
            # and row bbox[1] with length bbox[3]
            if (bbox[0] <= centroid[0] <= bbox[0] + bbox[2]) and \
               (bbox[1] <= centroid[1] <= bbox[1] + bbox[3]):

               # for the ith predicted centroid, which ground-truth box does it belong to
                centroid_idx[i] = j
                # calculates the number of centroids within the jth bbox
                bbox_hits[j] += 1
                break

    # Map each predicted centroid to the number of times it intersects with a
    # ground-truth labeled contour. This intersection occurs when a tomato is
    # split into multiple segments by e.g. a leaf or stem
    # ?? Not really? This finds the number of predicted centroids within
    # a single ground-truth box a
    totals = {int(a) : np.sum(centroid_idx == a) for a in np.unique(centroid_idx)}

    # True positives: A unique prediction on a uniquely-labeled tomato
    # totals[i] is the number of predicted centroids that fall within a single box
    # here a tp is only accepted when there is exactly one prediction of a single ground-truth tomato
    tp = np.sum(totals[key] == 1 for key in totals)

    # False positive: A prediction on a part of the image not labeled (ground-truthed) as tomato
    # Note that totals is a hash table so totals[-1] is the value of totals for key -1
    fp = totals[-1] if -1 in totals else 0

    # False negative: Number of labeled tomatoes that the predictions missed
    fn = np.sum(bbox_hits == 0)

    # "Discard" are instances where there are multiple predictions per tomato
    # num_discard is the number of predictions that have been discarded because
    # they predict the same tomatoes
    num_discard = np.sum(totals[key] if totals[key] > 1 and key != -1 else \
                         0 for key in totals)

    # "num_contour" are the number of tomatoes that have multiple predictions
    num_contour = np.sum(1 if totals[key] > 1 and key != -1 else \
                         0 for key in totals)



    # Use broadcasting to find the distance between all predicted centroids and
    # labeled centroids
    yp = yp_centroids[:, np.newaxis]
    yt = np.asarray(yt_centroids)[np.newaxis]
    dist = np.sqrt(np.sum((yp - yt) ** 2, axis=2))
    overall_dist = image_width #used to define a large value for distance when no true positives are found
    tp_dist = image_width #used to define a large value for distance when no true positives are found

    # Find the indices for the labeled centroids that we've found
    yt_hits = np.asarray([a for a in totals if totals[a] == 1])
    if yt_hits.size > 0:
        yp_idx = [np.argwhere(centroid_idx == a) for a in yt_hits]
        yp_idx = np.asarray(yp_idx).flatten()

        overall_dist = np.mean(np.min(dist, axis=1))
        tp_dist = np.mean(dist[yp_idx, yt_hits])
    """print 'tp=', tp, 'fp=',fp, 'fn=',fn, 'num_discard=',num_discard, 'num_contou=',num_contour, \
            'overall_dist=',overall_dist, 'tp_dist=', tp_dist"""
    return np.atleast_2d([tp, fp, fn, num_discard, num_contour,
                          overall_dist, tp_dist])

def validate_centroids_stem(yp_centroids,  yt_centroid_list, yt_contours):
    """Calculates how many centroids are predicted per ground-truth / labeled
    contour, and the distance from predicted to labeled centroids.
    """

    """print "centroids:", centroid[good_idx][:,0], type(centroid[good_idx]), centroid[good_idx].shape
    if len(centroid[good_idx])>1:
        tck, u = splprep([centroid[good_idx][:,0],centroid[good_idx][:,1]], k=min(3,len(centroid[good_idx])-1))
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)

        plt.imshow(mask.astype(np.uint8))
        plt.plot(centroid[good_idx][:,0], centroid[good_idx][:,1], 'ro')
        plt.plot(x_new, y_new, 'b--')
        plt.show()
    """
    centroid_idx = np.ones((len(yp_centroids),))*-1
    stem_hits = np.zeros((len(yt_centroid_list),))
    yt_stem = np.zeros((len(yt_centroid_list),))
    # Create ground-truth stem curves
    i=0
    for centroid_list in yt_centroid_list:
        if len(centroid_list)>1:
            # Create the spline representation of the stem. Returns the knot-points
            # tck, A tuple (t,c,k) containing the vector of knots, the B-spline coefficients,
            # and the degree of the spline
            # and  u, an array of parameter values.
            tck, u = splprep([centroid_list[:,0],centroid_list[:,1]], k=min(3,len(centroid_list)-1))
            #Sample points from u the at an interval of 1000
            u_new = np.linspace(u.min(), u.max(), 1000)
            # evaluate the spline at u_new given knot-points tck
            x_new, y_new = splev(u_new, tck, der=0)
            yt_stem[i].append([x_new, y_new])
        else:
            yt_stem[i].append(centroid_list)

    # for each predicted centroid, check whether we've hit a labeled object by seeing if
    # the prediction is close enough to a given stem curve
    for i, centroid in enumerate(yp_centroids):
        min_dist = None
        # go over all ground-truth lines until you find the curve closest to the centroid
        for j in enumerate(yt_stem):

            # Check if the current predicted centroid is within the perimeter of bbox.
            # bbox starts from y value (column) bbox[0] with width bbox[2]
            # and row bbox[1] with length bbox[3]
            if (bbox[0] <= centroid[0] <= bbox[0] + bbox[2]) and \
               (bbox[1] <= centroid[1] <= bbox[1] + bbox[3]):

               # for the ith predicted centroid, which ground-truth box does it belong to
                centroid_idx[i] = j
                # calculates the number of centroids within the jth bbox
                stem_hits[j] += 1
                break

    # Map each predicted centroid to the number of times it intersects with a
    # ground-truth labeled contour. This intersection occurs when a tomato is
    # split into multiple segments by e.g. a leaf or stem
    # ?? Not really? This finds the number of predicted centroids within
    # a single ground-truth box a
    totals = {int(a) : np.sum(centroid_idx == a) for a in np.unique(centroid_idx)}

    # True positives: A unique prediction on a uniquely-labeled tomato
    # totals[i] is the number of predicted centroids that fall within a single box
    # here a tp is only accepted when there is exactly one prediction of a single ground-truth tomato
    tp = np.sum(totals[key] == 1 for key in totals)

    # False positive: A prediction on a part of the image not labeled (ground-truthed) as tomato
    # Note that totals is a hash table so totals[-1] is the value of totals for key -1
    fp = totals[-1] if -1 in totals else 0

    # False negative: Number of labeled tomatoes that the predictions missed
    fn = np.sum(bbox_hits == 0)

    # "Discard" are instances where there are multiple predictions per tomato
    # num_discard is the number of predictions that have been discarded because
    # they predict the same tomatoes
    num_discard = np.sum(totals[key] if totals[key] > 1 and key != -1 else \
                         0 for key in totals)

    # "num_contour" are the number of tomatoes that have multiple predictions
    num_contour = np.sum(1 if totals[key] > 1 and key != -1 else \
                         0 for key in totals)



    # Use broadcasting to find the distance between all predicted centroids and
    # labeled centroids
    yp = yp_centroids[:, np.newaxis]
    yt = np.asarray(yt_centroids)[np.newaxis]
    dist = np.sqrt(np.sum((yp - yt) ** 2, axis=2))
    overall_dist = image_width #used to define a large value for distance when no true positives are found
    tp_dist = image_width #used to define a large value for distance when no true positives are found

    # Find the indices for the labeled centroids that we've found
    yt_hits = np.asarray([a for a in totals if totals[a] == 1])
    if yt_hits.size > 0:
        yp_idx = [np.argwhere(centroid_idx == a) for a in yt_hits]
        yp_idx = np.asarray(yp_idx).flatten()

        overall_dist = np.mean(np.min(dist, axis=1))
        tp_dist = np.mean(dist[yp_idx, yt_hits])
    print 'tp=', tp, 'fp=',fp, 'fn=',fn, 'num_discard=',num_discard, 'num_contou=',num_contour, \
            'overall_dist=',overall_dist, 'tp_dist=', tp_dist
    return np.atleast_2d([tp, fp, fn, num_discard, num_contour,
                          overall_dist, tp_dist])

def meanIOU_instances(preds, targets, image_name, instance_stats_for_class, instance_found_thresh=0.5):
    """Returns the mean intersection over union for each instance in the preds array of instance masks.

    Notes
    -----
    instance_found_thresh is the minimum value of iou to count the instance as having been recognized.
    """
    instance_stats_for_class[image_name]={'mIOU':0,\
                                     'found':False,\
                                     'num_preds':0,\
                                     'num_instances':0}
    iou_mask=[]
    if preds==[]:
        instance_stats_for_class[image_name]['num_instances']=len(targets)
    else:
        # Keep track of an array for each input prediction / target pair.
        # Also keep track of a "found" mask, so when we calculate the mean
        # it takes into account any instances where a class doesn't exist in image
        iou = np.zeros(len(targets))
        curr_intersection=np.zeros(preds.shape)
        curr_union = np.zeros(preds.shape)
        found = 0
        if len(targets)>0:
            iou_mask=np.zeros(targets[0].shape).astype(np.uint8)

            #highlight ground truth instances
            for i in xrange(len(targets)):
                iou_mask[targets[i]>0] = 200

            """plt.imshow(iou_mask)
            plt.title('mask with gt')
            plt.show()"""

            num_preds= len(np.unique(preds[preds!=0]))

            #highlight predicted instances
            iou_mask[preds!=0] =128
            """plt.imshow(preds)
            plt.title('preds'+str(num_preds))
            plt.show()"""
        for i in xrange(len(targets)):
            max_iou=0
            for j in np.unique(preds[preds!=0]):
                curr_intersection=np.zeros(preds.shape)
                curr_union = np.zeros(preds.shape)
                curr_union[(targets[i]>0)]=1
                curr_intersection[preds==j]=1
                curr_intersection= curr_intersection * curr_union
                curr_union[preds==j]=1
                curr_iou = float(np.sum(curr_intersection>0)) / (np.sum(curr_union > 0) + 1e-8)
                if curr_iou>max_iou:
                    max_iou=curr_iou
                    max_intersection = curr_intersection
                    max_union = curr_union
                    max_j=j

            iou[i] = max_iou

            """if max_iou==0:
                plt.imshow(targets[i])
                plt.title('could not find this')
                plt.show()"""

            if iou[i] > instance_found_thresh:
                found+=1
            if max_iou>0:
                iou_mask[(max_intersection > 0)]=255

        instance_stats_for_class[image_name]['num_instances']=len(targets)
        instance_stats_for_class[image_name]['num_preds']=num_preds
        instance_stats_for_class[image_name]['mIOU']=np.mean(iou)
        instance_stats_for_class[image_name]['found']=found

        """print "num_instances:{},num_pred:{},miou:{},found:{}".format(instance_stats_for_class[image_name]['num_instances'],\
                            instance_stats_for_class[image_name]['num_preds'],\
                            instance_stats_for_class[image_name]['mIOU'],\
                            instance_stats_for_class[image_name]['found'])"""
    return instance_stats_for_class, iou_mask


def summarize_instance_stats (classes_of_interest, test_keys, instance_stats, detailed=False):

    summ=dict.fromkeys(classes_of_interest, {})
    for _, curr_class in enumerate(classes_of_interest):
        summ[curr_class]=dict.fromkeys(['mean_avg_recall','mean_avg_precision',\
                                        'total_gt_instances','total_preds','total_found'])
        if len(instance_stats[curr_class].keys())>0:
            total_found =0
            total_gt_instances=0
            total_preds=0
            mean_avg_precision=0
            mean_avg_recall=0
            for key in test_keys:
                precision=0
                precentage_found=0
                if instance_stats[curr_class][key]['num_preds']>0:
                    precision = float(instance_stats[curr_class][key]['found'])/instance_stats[curr_class][key]['num_preds']
                if instance_stats[curr_class][key]['num_instances']>0:
                    precentage_found = float(instance_stats[curr_class][key]['found'])/instance_stats[curr_class][key]['num_instances']
                total_found += instance_stats[curr_class][key]['found']
                total_gt_instances += instance_stats[curr_class][key]['num_instances']
                total_preds += instance_stats[curr_class][key]['num_preds']
                mean_avg_precision += precision
                if instance_stats[curr_class][key]['num_instances']>0:
                    mean_avg_recall += float(instance_stats[curr_class][key]['found'])/instance_stats[curr_class][key]['num_instances']

            if total_gt_instances>0:
                precentage_found= float(total_found)/total_gt_instances
            if len(test_keys)>0:
                mean_avg_precision= float(mean_avg_precision)/len(test_keys)
                mean_avg_recall = mean_avg_recall/len(test_keys)

            summ[curr_class]['mean_avg_recall']=mean_avg_recall
            summ[curr_class]['mean_avg_precision']=mean_avg_precision
            summ[curr_class]['total_gt_instances']=total_gt_instances
            summ[curr_class]['total_preds']=total_preds
            summ[curr_class]['total_found']=total_found

    return summ

def display_instance_stats(classes_of_interest, test_keys, instance_stats, summ, detailed=False):

    if detailed==True:
        header= "Class\timage                       \trecall   \tprec   \t\tfound\t#instances\t#predictions"
    else:
        header= "Class\tmAR   \tmAP   \tfound\t#instances\t#predictions"
        print header

    for _, curr_class in enumerate(classes_of_interest):
        if detailed==True:
            if len(instance_stats[curr_class].keys())>0:
                print header
                total_found =0
                total_gt_instances=0
                total_preds=0
                mean_avg_precision=0
                mean_avg_recall=0
                for key in test_keys:
                    precision=0
                    precentage_found=0
                    if instance_stats[curr_class][key]['num_preds']>0:
                        precision = float(instance_stats[curr_class][key]['found'])/instance_stats[curr_class][key]['num_preds']
                    if instance_stats[curr_class][key]['num_instances']>0:
                        precentage_found = float(instance_stats[curr_class][key]['found'])/instance_stats[curr_class][key]['num_instances']
                    print "{:5}\t{:25}\t{:8.2f}\t{:8.2f}\t{:5}\t{:10}\t{}".format(curr_class, key, \
                                                                 precentage_found, \
                                                                 precision,\
                                                                 instance_stats[curr_class][key]['found'],\
                                                                 instance_stats[curr_class][key]['num_instances'],
                                                                 instance_stats[curr_class][key]['num_preds'])

                print "========================================"
                print "{:5}\t{:25}\tmAR:{:4.2f}\tmAP:{:4.2f}\t{:5}\t{:10}\t{}".format(curr_class, 'TOTAL', \
                                                        summ[curr_class]['mean_avg_recall'], \
                                                        summ[curr_class]['mean_avg_precision'],\
                                                        summ[curr_class]['total_found'], \
                                                        summ[curr_class]['total_gt_instances'],
                                                        summ[curr_class]['total_preds'])
        else:
            print "{:5}\t{:4.2f}\t{:4.2f}\t{:5}\t{:10}\t{}".format(curr_class, \
                                                        summ[curr_class]['mean_avg_recall'], \
                                                        summ[curr_class]['mean_avg_precision'],\
                                                        summ[curr_class]['total_found'], \
                                                        summ[curr_class]['total_gt_instances'],
                                                        summ[curr_class]['total_preds'])

def get_stats_per_experiment(classes_of_interest,summ_stats):

    exp_summs=dict.fromkeys(classes_of_interest)
    for curr_class in classes_of_interest:
        exp_summs[curr_class]=dict.fromkeys(['mean_avg_recall','mean_avg_precision',\
                                        'total_gt_instances','total_preds','total_found'], 0.0)
        num_runs= len(summ_stats)
        for j in xrange(0, num_runs):
            exp_summs[curr_class]['mean_avg_recall']+=summ_stats[j][curr_class]['mean_avg_recall']
            exp_summs[curr_class]['mean_avg_precision']+=summ_stats[j][curr_class]['mean_avg_precision']
            exp_summs[curr_class]['total_found']+=summ_stats[j][curr_class]['total_found']
            exp_summs[curr_class]['total_gt_instances']+=summ_stats[j][curr_class]['total_gt_instances']
            exp_summs[curr_class]['total_preds']+=summ_stats[j][curr_class]['total_preds']
        if num_runs>0:
            exp_summs[curr_class]['mean_avg_recall']=float(exp_summs[curr_class]['mean_avg_recall'])/num_runs
            exp_summs[curr_class]['mean_avg_precision']=float(exp_summs[curr_class]['mean_avg_precision'])/num_runs
            exp_summs[curr_class]['total_found']=float(exp_summs[curr_class]['total_found'])/num_runs
            exp_summs[curr_class]['total_gt_instances']=float(exp_summs[curr_class]['total_gt_instances'])/num_runs
            exp_summs[curr_class]['total_preds']=float(exp_summs[curr_class]['total_preds'])/num_runs

    return exp_summs

def display_overall_exp_stats(total_summs, classes_of_interest):

    header= "Exp\t\tClass\tmAR   \tmAP   \tfound\t#instances\t#predictions"
    print header
    for experiment in total_summs.keys():
        for curr_class in classes_of_interest:
            print "{:25}\t{:5}\t{:4.2f}\t{:4.2f}\t{:5}\t{:10}\t{}".format(experiment, curr_class, \
                                                        total_summs[experiment][curr_class]['mean_avg_recall'], \
                                                        total_summs[experiment][curr_class]['mean_avg_precision'],\
                                                        total_summs[experiment][curr_class]['total_found'], \
                                                        total_summs[experiment][curr_class]['total_gt_instances'],
                                                        total_summs[experiment][curr_class]['total_preds'])



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
