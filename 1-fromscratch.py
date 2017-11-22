from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from matplotlib.lines import Line2D

# seed the random number generator
np.random.seed(7)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Loading data
X = scipy.io.loadmat('06_Data_Bidimensional.mat')['X']

# Generating "random" centroids
centroids = np.array([[3,3], [6, 2], [8, 5]], dtype=np.float32)

# To store the value of centroids when it updates
centr_old = np.zeros(centroids.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(centroids, centr_old, None)

# Save for later
centroids_history = []

# number of clusters
k = 3

# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    # Storing the old centroid values
    centr_old = deepcopy(centroids)

    # Saving for the lines
    centroids_history.append(deepcopy(centroids))

    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    error = dist(centroids, centr_old, None)

colors = [ 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

ax.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=50, c='r')

# Plot lines
for i in range(len(centroids_history) - 1):
    for j in range(k):
        ax.plot([centroids_history[i][j][0], centroids_history[i+1][j][0]], [centroids_history[i][j][1], centroids_history[i+1][j][1]], marker="x", c='k')


plt.show()
