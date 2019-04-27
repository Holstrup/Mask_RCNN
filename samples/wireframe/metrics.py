import sklearn.cluster as skclus
from sklearn import metrics
import pandas as pd
import sklearn
from samples.wireframe.database_actions import get_known_encodings

embeddings, labels = get_known_encodings("Database.db", 128)

def metricFunction(embeddings, labels, n_clusters):
    algorithms = []
    algorithms.append(skclus.KMeans(n_clusters=n_clusters, random_state=1))
    algorithms.append(sklearn.cluster.MeanShift())
    algorithms.append(sklearn.cluster.DBSCAN())
    algorithms.append(sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters))
    algorithms.append(sklearn.cluster.AffinityPropagation())

    data = []
    for algo in algorithms:
        algo.fit(embeddings.T)
        data.append(({
            'ARI': metrics.adjusted_rand_score(labels, algo.labels_),
            'AMI': metrics.adjusted_mutual_info_score(labels, algo.labels_),
            'Homogenity': metrics.homogeneity_score(labels, algo.labels_),
            'Completeness': metrics.completeness_score(labels, algo.labels_),
            'V-measure': metrics.v_measure_score(labels, algo.labels_),
            'Silhouette': metrics.silhouette_score(embeddings.T, algo.labels_)}))

    results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                               'Completeness', 'V-measure',
                                               'Silhouette'],
                           index=['K-means', 'MeanShift', 'DBSCAN', 'Agglomerative', 'AffinityPropagation'])
    return results