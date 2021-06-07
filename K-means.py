import numpy as np
import matplotlib.pyplot as plt

#Find Euclidean Distance

def euclideanDistance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KMeans():

    def __init__(self, K=5, maxIters=100, shouldPlotSteps=False):
        self.K = K
        self.maxIters = maxIters
        self.shouldPlotSteps = shouldPlotSteps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.samplesCount, self.featuresCount = X.shape

        initialRandomIdxs = np.random.choice(
            self.samplesCount, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in initialRandomIdxs]

        for _ in range(self.maxIters):
            self.clusters = self.createClusters(self.centroids)

            if self.shouldPlotSteps:
                self.plot()

            oldCentroids = self.centroids
            self.centroids = self.getCentroids(self.clusters)

            if self.isConverged(oldCentroids, self.centroids):
                break

            if self.shouldPlotSteps:
                self.plot()

        return self.getClusterLabels(self.clusters)

    def getClusterLabels(self, clusters):
        labels = np.empty(self.samplesCount)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def createClusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.get_closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def get_closest_centroid(self, sample, centroids):
        distances = [euclideanDistance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def getCentroids(self, clusters):
        centroids = np.zeros((self.K, self.featuresCount))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def isConverged(self, centroids_old, centroids):
        distances = [euclideanDistance(
            centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point, linewidth=2,)

        for point in self.centroids:
            ax.scatter(*point, marker="x", linewidth=20)

        plt.show()


attributes = []
m_class = []

if __name__ == "__main__":
  with open('dataset/iris.data') as file:   
    data = file.read()
    lines = data.split('\n')
    for line in lines:
      row = line.split(',')
      if len(row) > 1:
        features = list(map(float, row[:-1]))
        label = row[-1]
        attributes.append(features)
        m_class.append(label)

    model = KMeans(K=len(np.unique(m_class)), shouldPlotSteps=False)
    X = np.array(attributes, np.float64)
    model.predict(X)
    model.plot()
