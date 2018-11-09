import os
import numpy as np
import cv2

image_width = 860

def merge_segmented_tomatoes(segmentation, stats, dist_thresh=40,
                             join_area_thresh=400):

    # Convert values to [0, 255] then calculate hough circles
    gray = segmentation.astype(np.uint8)
    gray[gray > 0] = 255

    merged_indices, new_centroid, new_bbox = [], [], []

    # Calculate distances between all contours and all other contours
    # If distance between two very small contours is low, merge them
    diff = stats['centroid'][np.newaxis] - stats['centroid'][:, np.newaxis]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    area_mask = np.atleast_2d(stats['bbox'][:, -1] < join_area_thresh)
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


if __name__ == '__main__':
    main()
