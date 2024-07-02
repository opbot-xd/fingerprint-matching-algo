import cv2
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sample = cv2.imread('SOCOFing/Altered/Altered-Medium/10__M_Left_little_finger_CR.BMP')

# Detect Harris corners
gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
gray_sample = np.float32(gray_sample)
harris_corners_sample = cv2.cornerHarris(gray_sample, 2, 3, 0.04)
harris_corners_sample = cv2.dilate(harris_corners_sample, None)

# Threshold to find strong corners
threshold = 0.01 * harris_corners_sample.max()
kp1 = [cv2.KeyPoint(float(x), float(y), 1) for y, x in zip(*np.where(harris_corners_sample > threshold))]
orb = cv2.ORB_create()
kp1, des1 = orb.compute(sample, kp1)

def compute_keypoints_descriptors(file):
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
    
    gray_fingerprint_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
    gray_fingerprint_image = np.float32(gray_fingerprint_image)
    harris_corners = cv2.cornerHarris(gray_fingerprint_image, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    
    threshold = 0.01 * harris_corners.max()
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for y, x in zip(*np.where(harris_corners > threshold))]
    kp2, des2 = orb.compute(fingerprint_image, kp2)
    
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2

def knn_match(args):
    fil, kp2_info, des2 = args
    
    kp2 = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) 
           for (x, y), _size, _angle, _response, _octave, _class_id in kp2_info]
    
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,    # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=1)   # or pass empty dictionary
    matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(des1, des2, k=2)
    match_point = []
    for p_q in matches:
        if len(p_q) == 2:
            p, q = p_q
            if p.distance < 0.3 * q.distance:
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

    all_kp2_des2.extend(all_kp2_des2)
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
