import cv2
import os

sample = cv2.imread('SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP')

image = None
filename = None 
import time
best_score = 0
kp1, kp2, mp = None, None, None
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(sample, None)
# caLculate the time
# Loop through all the files in the Real folder

# end timer
all_kp2_des2 = []
process_time = time.time()

# Loop through all the files in the Real folder and compute keypoints and descriptors
for file in os.listdir('SOCOFing/Real/'):
    # Tho we will pre-compute it
    start_single_image = time.time()
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))

    # Detect keypoints and compute descriptors for each image
    kp2, des2 = sift.detectAndCompute(fingerprint_image, None)
    end_single_image = time.time()
    print('Processing:', file, 'Time:', end_single_image-start_single_image)

    all_kp2_des2.append((file, kp2, des2))


end_process_time = time.time()

# duplicate the all_kp2_des2
all_kp2_des2.extend(all_kp2_des2)
# all_kp2_des2.extend(all_kp2_des2)


# Start timer
knn_start = time.time()
counter = 0
# Loop through all the precomputed keypoints and descriptors
for fil, kp2, des2 in all_kp2_des2:
    # Perform matching between sample and current image
    single_matching_start = time.time()
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(des1, des2, k=2)

    match_point = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_point.append(p)

    # Calculate the number of keypoints for scoring
    keypoints = min(len(kp1), len(kp2))

    # Calculate the score for the current image
    score = len(match_point) / keypoints * 100

    # Update the best match if the current score is higher
    if score > best_score:
        best_score = score
        image = fingerprint_image
        filename = fil
        kp2 = kp2
        mp = match_point
    counter += 1
    single_matching_end = time.time()
    print('Matching:', fil, 'Time:', single_matching_end-single_matching_start)

# End timer
knn_end = time.time()

print('Best Score:', best_score)
print('Filename:', filename)
print('Knn Time:', knn_end-knn_start)
print('Single Knn Time:', single_matching_end - single_matching_start)
print('Image Process Time:', end_process_time- process_time)


