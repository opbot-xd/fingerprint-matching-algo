import cv2
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Read the sample fingerprint image
sample = cv2.imread('SOCOFing/Altered/Altered-Easy/5__M_Left_little_finger_CR.BMP',flags = cv2.IMREAD_GRAYSCALE)

# Create AKAZE detector
akaze = cv2.AKAZE_create()

# Detect keypoints and compute descriptors for the sample image
kp1, des1 = akaze.detectAndCompute(sample, None)
des1 = np.float32(des1)  # Convert descriptors to float32

def compute_keypoints_descriptors(file):
    # Read a fingerprint image
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file),flags=cv2.IMREAD_GRAYSCALE)
    # Detect keypoints and compute descriptors using AKAZE
    kp2, des2 = akaze.detectAndCompute(fingerprint_image, None)
    if des2 is not None:
        des2 = np.float32(des2)  # Convert descriptors to float32
    else:
        des2 = np.array([], dtype=np.float32)  # Ensure des2 is an empty array if no descriptors are found
    # Store keypoint information for later reconstruction
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2

def knn_match(args):
    fil, kp2_info, des2 = args

    # Reconstruct keypoints from the stored information
    kp2 = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) 
           for (x, y), _size, _angle, _response, _octave, _class_id in kp2_info]

    if len(des2) < 2:  # Skip matching if there are fewer than 2 descriptors
        return fil, 0

    # FLANN parameters for AKAZE
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=10)
    search_params = {'checks':50}

    # Use FlannBasedMatcher to match descriptors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.8
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Calculate the matching score
    keypoints = min(len(kp1), len(kp2))
    score = len(good_matches) / keypoints * 100

    return fil, score

if __name__ == '__main__':
    process_time = time.time()

    # Compute keypoints and descriptors for all fingerprint images in the dataset
    with Pool(cpu_count()) as p:
        all_kp2_des2 = p.map(compute_keypoints_descriptors, os.listdir('SOCOFing/Real/'))

    end_process_time = time.time()

    # Duplicate the data to make the size 18000
    all_kp2_des2.extend(all_kp2_des2)

    best_score = 0
    best_match = None
    knn_start = time.time()

    # Perform knn matching in parallel
    with Pool() as p:
        print('Number of CPU:', cpu_count())
        results = p.map(knn_match, all_kp2_des2)

    knn_end = time.time()
    print("Number of fingers we ran search on: ", len(results))

    # Find the best match
    loop_time_start = time.time()
    for fil, score in results:
        if score > best_score:
            best_score = score
            best_match = fil
    loop_time_end = time.time()

    # Print results
    print('Best Score:', best_score)
    print('Filename:', best_match)
    print('Knn Time (Actual time that machine will take):', knn_end - knn_start)
    print('Image Process Time (We will pre-compute it):', end_process_time - process_time)
    print('Loop Time:', loop_time_end - loop_time_start)
    print('Total Time:', time.time() - process_time)
