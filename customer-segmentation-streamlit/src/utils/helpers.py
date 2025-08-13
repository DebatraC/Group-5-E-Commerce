def format_user_clusters(user_clusters):
    """Format user clusters for display."""
    return user_clusters.rename(columns={'user_id': 'User ID', 'visit_count': 'Visit Count', 'cluster': 'Cluster'})

def format_cluster_analysis(cluster_analysis):
    """Format cluster analysis results for display."""
    return cluster_analysis.round(2)

def get_cluster_profile(cluster_data):
    """Generate a profile for each cluster based on its characteristics."""
    profiles = []
    for cluster_id, data in cluster_data.groupby('cluster'):
        avg_visits = data['visit_frequency'].mean()
        avg_ratio = data['view_to_action_ratio'].mean()
        avg_brands = data['brand_interactions'].mean()
        
        profile = {
            'Cluster ID': cluster_id,
            'Average Visits': avg_visits,
            'Average View-to-Action Ratio': avg_ratio,
            'Average Brand Interactions': avg_brands
        }
        profiles.append(profile)
    return profiles

def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd
    return pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path)

def save_data(data, file_path):
    """Save data to a specified file path."""
    if file_path.endswith('.parquet'):
        data.to_parquet(file_path, index=False)
    else:
        data.to_csv(file_path, index=False)