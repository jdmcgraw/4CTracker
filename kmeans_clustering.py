import numpy as np
from random import randint, sample, seed


class KmeansClassifier:

    def __init__(self, data, clusters, min_iter=1000):
        self.rgbd_data = np.array(data)
        self.clusters = min(clusters, len(data))

        self.num_frames = self.rgbd_data.shape[0]
        self.rgbd_flat = np.reshape(self.rgbd_data, (self.num_frames, -1))
        self.data_dimension = self.rgbd_flat.shape[1]
        self.cluster_dictionary = {}  # data index : cluster index
        self.min_iter = min_iter

        seed(0)
        init_centroids = sample(list(self.rgbd_flat), self.clusters)
        self.centroids = np.array(init_centroids)

        self.converged, i = False, 0
        while not self.converged and i < self.min_iter:
            self.compute_closest_centroids()
            self.recompute_centroid_positions()
            i += 1

        self.output = []
        for i in range(clusters):
            cluster_i = [k for k, v in self.cluster_dictionary.items() if v == i]
            self.output.append(cluster_i)

    def get_clusters(self):
        return [k[randint(0, len(k)-1)] for k in self.output]

    def compute_closest_centroids(self):
        for i, point in enumerate(self.rgbd_flat):
            min_distance = float('inf')
            for j, cluster in enumerate(self.centroids):
                dist = np.linalg.norm(point-cluster)
                if dist < min_distance:
                    self.cluster_dictionary[i] = j
                    min_distance = dist

    def recompute_centroid_positions(self):
        self.converged = True
        for i in range(self.clusters):
            points = [k for k, v in self.cluster_dictionary.items() if v == i]
            data = [self.rgbd_flat[point] for point in points]
            if points:
                self.centroids[i] = np.mean(data)
            else:
                self.converged = False
