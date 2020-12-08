"""
Clusters the bounding boxes based on their widths and finds max of each
cluster.
"""

import numpy as np
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt


NETWORK_HEIGHT = 64

boxes = np.load('data.npy')
widths = np.array([])
for box in boxes:
    width, height = np.abs(box[1] - box[0])
    width = NETWORK_HEIGHT / height * width
    widths = np.append(widths, width)

print(np.max(widths))
cluster_centroids, labels = vq.kmeans2(widths, 3, iter=1000)

cluster_max = {}
for idx, label in enumerate(labels):
    if label not in cluster_max:
        cluster_max[label] = 0
    cluster_max[label] = widths[idx] if widths[idx] > cluster_max[label] else cluster_max[label]

for label in cluster_max:
    print('clusterMax[{}]: {}'.format(label, cluster_max[label]))
