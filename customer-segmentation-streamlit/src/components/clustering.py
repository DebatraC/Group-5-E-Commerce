from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

def perform_clustering(user_visit_counts, n_clusters=4):
    scaler = StandardScaler()
    visit_counts_scaled = scaler.fit_transform(user_visit_counts.values.reshape(-1, 1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(visit_counts_scaled)

    user_clusters = pd.DataFrame({
        'user_id': user_visit_counts.index,
        'visit_count': user_visit_counts.values,
        'cluster': clusters
    })

    return user_clusters

def analyze_clusters(user_clusters):
    return user_clusters.groupby('cluster')['visit_count'].describe()