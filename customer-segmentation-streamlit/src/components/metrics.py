def calculate_silhouette_score(X, labels):
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels)

def calculate_inertia(X, labels):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(set(labels)), random_state=42)
    kmeans.fit(X)
    return kmeans.inertia_

def display_metrics(X, labels):
    silhouette = calculate_silhouette_score(X, labels)
    inertia = calculate_inertia(X, labels)
    
    metrics = {
        "Silhouette Score": silhouette,
        "Inertia": inertia
    }
    
    return metrics

def show_metrics(metrics):
    import streamlit as st
    
    st.subheader("Clustering Metrics")
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")