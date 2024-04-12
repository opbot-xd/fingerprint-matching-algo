# fingerprint-matching-algo


Dataset on https://www.kaggle.com/datasets/ruizgara/socofing?resource=download


After Parallelization (Total Fingerprint 12,000)
```
Best Score: 63.63636363636363
Filename: 1__M_Left_little_finger.BMP
Knn Time: 2.4496026039123535
Image Process Time: 3.2450079917907715 (for only 6k)
Total Time: 5.695852518081665
```


Last Benchmark (Total Fingerprint 12,000)
```
Best Score: 63.63636363636363
Filename: 1__M_Left_little_finger.BMP
Knn Time: 17.922457218170166
Single Knn Time: 0.0011396408081054688
Image Process Time: 14.543917417526245 (for only 6k)
```




Ways to optimize
1. Use Faster Feature Detector: Instead of SIFT, you can use faster feature detectors like ORB or AKAZE, which are generally faster than SIFT while still providing decent performance.
2. Limit Keypoint Detection: You can limit the number of keypoints detected in both the sample and target images. Since you are comparing fingerprints, not all keypoints are necessary for matching.
3. Parallelization: Use parallel processing to perform matching on multiple images simultaneously, which can significantly reduce the overall processing time, especially if you have a multi-core CPU.
4. Approximate Nearest Neighbors: Use approximate nearest neighbor search algorithms like FLANN or KD-trees, which are faster than exact nearest neighbor search.
5. Optimize Matching Parameters: Adjust matching parameters such as the distance ratio threshold and the number of trees in the FLANN index to improve matching speed without sacrificing accuracy.
6. Precompute Keypoints and Descriptors
7. Reduce Image Size