from typing import Union
from fastapi import FastAPI


import cv2
import os
import time
import concurrent.futures
from multiprocessing import Pool, cpu_count
import psutil
# import cProfile
# import re
# cProfile.run('re.compile("foo|bar")')


sample = cv2.imread('SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP')
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(sample, None)

def compute_keypoints_descriptors(file):
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
    kp2, des2 = sift.detectAndCompute(fingerprint_image, None)
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2

all_kp2_des2 = []

# with Pool(cpu_count()) as p:
for file in os.listdir('SOCOFing/Real/'):
    print(file)
    all_kp2_des2.append(compute_keypoints_descriptors(file))
    # all_kp2_des2=  p.map(compute_keypoints_descriptors, os.listdir('SOCOFing/Real/'))




# iterate all the images in SOCOFing/Real/ and compute the keypoints and descriptors
# all_kp2_des2 = []
#     fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
#     kp2, des2 = sift.detectAndCompute(fingerprint_image, None)
#     kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
#     all_kp2_des2.append((file, kp2_info, des2))



def knn_match(args):
    fil, kp2_info, des2 = args
    
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


def dmain():
#if __name__ == '__main__':
    process_time = time.time()


    #NOTE kp1 is list of keypoints and des is numpy array of shape (number_of_keypoints, 128)
    # ggdf = cv2.drawKeypoints(sample, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('keypoints.jpg', ggdf) // to draw the keypoints of the fingerprint image




    end_process_time = time.time()
    # gg = all_kp2_des2
    # all_kp2_des2.extend(all_kp2_des2)
    # all_kp2_des2.extend(all_kp2_des2)
    # all_kp2_des2.extend(gg)
    best_score = 0
    best_match = None
    knn_start = time.time()

    # with concurrent.futures.ProcessPoolExecutor() as p:
    # count number of process
    
    with Pool() as p:
        print('Number of CPU:', cpu_count())
        results = p.map(knn_match, all_kp2_des2)

    knn_end = time.time()
    print("Number of fingers we ran search on: ",len(results))
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


app = FastAPI()

@app.get("/")
def read_root():
    dmain()
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    dmain()
    return {"item_id": item_id, "q": q}


