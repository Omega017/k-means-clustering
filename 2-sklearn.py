import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Loading data
X = scipy.io.loadmat('06_Data_Bidimensional.mat')['X']

k = 3
km = KMeans(n_clusters=k)
km.fit(X)

labels = km.labels_
centroids = km.cluster_centers_

colors = [ 'y', 'b', 'g']

for i in range(k):
    # select only data observations with cluster label == i
    ds = X[np.where(labels==i)]
    # plot the data observations
    plt.scatter(ds[:,0],ds[:,1], s=7, c=colors[i])
    # plot the centroids
    lines = plt.scatter(centroids[i,0],centroids[i,1], marker='o', s=50, c='r')
plt.show()
