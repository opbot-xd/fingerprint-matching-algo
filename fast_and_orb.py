import cv2
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Load the sample fingerprint image
sample = cv2.imread('SOCOFing/Altered/Altered-Medium/16__M_Left_little_finger_CR.BMP')

# Convert to grayscale
gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# Detect FAST keypoints
fast = cv2.FastFeatureDetector_create()
kp1 = fast.detect(gray_sample, None)

# Compute ORB descriptors for the detected keypoints
orb = cv2.ORB_create()
kp1, des1 = orb.compute(gray_sample, kp1)

def compute_keypoints_descriptors(file):
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
    gray_fingerprint_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
    
    # Detect FAST keypoints
    fast = cv2.FastFeatureDetector_create()
    kp2 = fast.detect(gray_fingerprint_image, None)
    
    # Compute ORB descriptors for the detected keypoints
    orb = cv2.ORB_create()
    kp2, des2 = orb.compute(gray_fingerprint_image, kp2)
    
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2

def knn_match(args):
    fil, kp2_info, des2 = args
    kp2 = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) 
           for (x, y), _size, _angle, _response, _octave, _class_id in kp2_info]
    
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=1)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)
    
    match_point = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.30 * n.distance:
                match_point.append(m)

    keypoints = min(len(kp1), len(kp2))
    score = len(match_point) / keypoints * 100 if keypoints > 0 else 0

    return fil, score

if __name__ == '__main__':
    process_start_time = time.time()

    with Pool(cpu_count()) as p:
        all_kp2_des2 = p.map(compute_keypoints_descriptors, os.listdir('SOCOFing/Real/'))

    best_score = 0
    best_match = None

    knn_start_time = time.time()

    with Pool() as p:
        results = p.map(knn_match, all_kp2_des2)

    knn_end_time = time.time()

    for fil, score in results:
        if score > best_score:
            best_score = score
            best_match = fil

    total_time = time.time() - process_start_time

    print('Best Score:', best_score)
    print('Filename:', best_match)
    print('KNN Time:', knn_end_time - knn_start_time)
    print('Total Time:', total_time)
