import cv2
import os
import time
from multiprocessing import Pool, cpu_count

def compute_keypoints_descriptors(file):
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
    kp2, des2 = sift.detectAndCompute(fingerprint_image, None)
    # Extract only the necessary information from KeyPoint objects
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2

def knn_match(args):
    fil, kp2_info, des2 = args
    
    # Reconstruct KeyPoint objects from extracted information
    kp2 = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) 
           for (x, y), _size, _angle, _response, _octave, _class_id in kp2_info]
    
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(des1, des2, k=2)
    
    match_point = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_point.append(p)

    keypoints = min(len(kp1), len(kp2))
    score = len(match_point) / keypoints * 100

    return fil, score

if __name__ == '__main__':
    sample = cv2.imread('SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP')
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(sample, None)

    process_time = time.time()
    all_kp2_des2 = []

    # Parallel computation of keypoints and descriptors
    with Pool(cpu_count()) as p:
        all_kp2_des2 = p.map(compute_keypoints_descriptors, os.listdir('SOCOFing/Real/'))

    end_process_time = time.time()
    all_kp2_des2.extend(all_kp2_des2)
    best_score = 0
    best_match = None
    knn_start = time.time()

    # Parallel computation of knn match
    with Pool(cpu_count()) as p:
        results = p.map(knn_match, all_kp2_des2)

    knn_end = time.time()

    for fil, score in results:
        if score > best_score:
            best_score = score
            best_match = fil

    print('Best Score:', best_score)
    print('Filename:', best_match)
    print('Knn Time:', knn_end - knn_start)
    print('Image Process Time:', end_process_time - process_time)
    print('Total Time:', time.time() - process_time)
