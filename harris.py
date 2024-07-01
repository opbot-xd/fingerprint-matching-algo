import cv2
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

sample = cv2.imread('SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP')

def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    keypoints = np.argwhere(dst > threshold * dst.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]
    return keypoints

# Harris corners on the sample image
kp1 = harris_corner_detection(sample)
sift = cv2.SIFT_create()
kp1, des1 = sift.compute(sample, kp1)

def compute_keypoints_descriptors(file):
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
    kp2 = harris_corner_detection(fingerprint_image)
    kp2, des2 = sift.compute(fingerprint_image, kp2)
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2

def knn_match(args):
    fil, kp2_info, des2 = args
    
    kp2 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), float(size), float(angle), float(response), int(octave), int(class_id))
           for (pt, size, angle, response, octave, class_id) in kp2_info]
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    match_point = []
    for p_q in matches:
        if len(p_q) == 2:
            p, q = p_q
            if p.distance < 0.75 * q.distance:
                match_point.append(p)

    keypoints = min(len(kp1), len(kp2))
    score = len(match_point) / keypoints * 100

    return fil, score

if __name__ == '__main__':
    process_time = time.time()

    all_kp2_des2 = []
    with Pool(cpu_count()) as p:
        all_kp2_des2 = p.map(compute_keypoints_descriptors, os.listdir('SOCOFing/Real/'))

    end_process_time = time.time()

    # Duplicate the data to make the dataset size match your requirement (18000 entries)
    all_kp2_des2.extend(all_kp2_des2)
    all_kp2_des2.extend(all_kp2_des2[:len(all_kp2_des2) // 3])

    best_score = 0
    best_match = None
    knn_start = time.time()

    with Pool() as p:
        print('Number of CPU:', cpu_count())
        results = p.map(knn_match, all_kp2_des2)

    knn_end = time.time()
    print("Number of fingers we ran search on: ", len(results))
    loop_time_start = time.time()
    for fil, score in results:
        if score > best_score:
            best_score = score
            best_match = fil
    loop_time_end = time.time() 

    print('Best Score:', best_score)
    print('Filename:', best_match)
    print('Knn Time (Actual time that machine will take):', knn_end - knn_start)
    print('Image Process Time (We will pre-compute it):', end_process_time - process_time)
    print('Loop Time:', loop_time_end - loop_time_start)
    print('Total Time:', time.time() - process_time)
