import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

def query(char):
    global words
    lst = []
    for i in words:
        if (char in i): lst.append(i)
    return(lst)


def plot_dendro(cluster_data, method='average', metric='cosine', figsize=[20,20]):
    plt.figure(figsize=(figsize[0], figsize[1]))
    dend = shc.dendrogram(shc.linkage(cluster_data, method='average', metric='cosine'))


def cluster(cluster_data, n_clusters=5):

    # Cluster algorithm
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
    cluster.fit_predict(cluster_data)

    # Print cluster label with word
    clustered = pd.DataFrame(list(zip(list(cluster_data.index), cluster.labels_)),
                             columns=['word', 'cluster'])

    for i in range(n_clusters):
        idx = clustered['cluster'] == i
        print(clustered[idx], '\n\n')
    
    # Return cluster object
    return(cluster)


# Print all words in a single cluster
def print_cluster(cluster_idx, cluster_data, cluster_obj):

    clustered = pd.DataFrame(list(zip(list(cluster_data.index), cluster_obj.labels_)),
                             columns=['word', 'cluster'])

    idx = clustered['cluster'] == cluster_idx
    print(clustered[idx].to_string())


# Return Mean Vector for each cluster (stored in a nested list)
def cluster_embeddings(cluster_data, cluster_labels):
    lst = []
    for i in list(set(cluster_labels)):
        mean_vector = cluster_data[cluster_labels == i].mean()
        lst.append(list(mean_vector))

    return(lst)

# Cosine similarity
def cos_sim(lst1, lst2, msg=''):
    cossim = dot(lst1, lst2)/(norm(lst1)*norm(lst2))
    print(msg, cossim)
    return(cossim)