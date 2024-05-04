import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from numpy.linalg import norm
import array

# randomly chooses k desired number of clusters by taking an input k and a dataset
def initialize(K, D):
  m,n = np.shape(D)
  centroids = np.empty((K, n))
  for i in range(K):
    centroids[i] = D[np.random.choice(range(m))]
  return centroids

# calculates the euclidean distance between two points
def distance(x1, x2):
  return np.linalg.norm(x1 - x2)

# finds nearest centroid for each data point
def chooseCentroid(x, C, K):
  distances = np.empty(K)
  for i in range(K):
    distances[i] = distance(C[i], x)
  return np.argmin(distances)

# assigns data point to a cluster, which is defined by an index
def createClusters(C, K, D):
  m = np.shape(D)[0]
  cIndex = np.empty(m)
  for i in range(m):
    cIndex[i] = chooseCentroid(D[i], C, K)
  return cIndex

# calculates the mean of the data points in each cluster
def findMean(cIndex, K, D):
  n = np.shape(D)[1]
  C = np.empty((K, n))
  for i in range(K):
    points = D[cIndex == i]
    C[i] = np.mean(points, axis=0)
  return C

# combine everything into Lloyd's algorithm
def lloyds(K, D, maxT = 500, tol = 10e-14):
  result = []
  C = initialize(K, D)
  print(f"initial centroids: {C}")
  for _ in range(maxT):
    clusters = createClusters(C, K, D)
    prevC = C
    C = findMean(clusters, K, D)
    difference = prevC - C
    if difference.all() < 0.05:
      print(f"final centroids: {C}")
      result.append(clusters)
      result.append(C)
      return result
  return result

if __name__=="__main__":
    # generate clustered dataset
    D, y = datasets.make_blobs()
    y_idx = lloyds(3, D)[0]
    print(y_idx[3])

    # plot actual dataset clusters
    fig,ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Actual Clusters")
    ax.scatter(D[:,0], D[:,1], c=y, cmap='viridis')
    plt.show()

    # plots clusters according to Lloyd's algorithm
    # **note: not always accurate, depends on initial centroids chosen
    fig,ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Lloyd's Algorithm")
    colors = []
    for i in range(len(y_idx)):
        if y_idx[i] == 0:
            ax.scatter(D[i][0],D[i][1], color="indigo")
            # colors.append("purple")
        elif y_idx[i] == 1:
            ax.scatter(D[i][0],D[i][1], color="gold")
            # colors.append("purple")
        elif y_idx[i] == 2:
            ax.scatter(D[i][0],D[i][1], color="teal")
            # colors.append("green")
    plt.show()