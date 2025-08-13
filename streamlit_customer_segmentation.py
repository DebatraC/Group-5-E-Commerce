import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-profile {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the customer data"""
    try:
        # Try to load the data
        data_path = './mytestdata.parquet'
        customer_data = pd.read_parquet(data_path)
        
        # Data preprocessing
        def fill_missing_brand(group):
            if group['brand'].notna().any():
                group['brand'] = group['brand'].fillna(method='ffill').fillna(method='bfill')
            return group
        
        customer_data = customer_data.groupby('product_id').apply(fill_missing_brand).reset_index(drop=True)
        customer_data = customer_data.dropna()
        
        return customer_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_user_metrics(customer_data):
    """Calculate user-level metrics for clustering"""
    # Group by user_id and calculate metrics
    user_metrics = customer_data.groupby('user_id').agg({
        'event_type': ['count'],  # Total interactions (visit frequency)
        'brand': 'nunique',       # Number of unique brands
        'price': 'mean'          # Average price per user
    }).reset_index()
    
    # Flatten column names
    user_metrics.columns = ['user_id', 'visit_frequency', 'brand_interactions', 'avg_price']
    
    # Calculate event type counts per user
    event_counts = customer_data.groupby(['user_id', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Merge with user metrics
    user_features = user_metrics.merge(event_counts, on='user_id', how='left')
    
    # Calculate view-to-action ratio: (cart + purchase) / view
    user_features['view_to_action_ratio'] = np.where(
        user_features['view'] > 0,
        (user_features.get('cart', 0) + user_features.get('purchase', 0)) / user_features['view'],
        0
    )
    
    return user_features

def perform_clustering(features, n_clusters, feature_cols):
    """Perform K-means clustering on selected features"""
    X = features[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, scaler, kmeans

def find_optimal_clusters(features, feature_cols, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    X = features[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        
        if len(set(clusters)) > 1:
            sil_score = silhouette_score(X_scaled, clusters, sample_size=min(10000, len(X_scaled)))
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
    
    return k_range, inertias, silhouette_scores

def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 E-Commerce Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    customer_data = load_data()
    if customer_data is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["Data Overview", "Visit-Based Clustering", "Price-Based Clustering", 
         "Multi-Feature Clustering", "Advanced Behavior Analysis"]
    )
    
    if page == "Data Overview":
        data_overview(customer_data)
    elif page == "Visit-Based Clustering":
        visit_based_clustering(customer_data)
    elif page == "Price-Based Clustering":
        price_based_clustering(customer_data)
    elif page == "Multi-Feature Clustering":
        multi_feature_clustering(customer_data)
    elif page == "Advanced Behavior Analysis":
        advanced_behavior_analysis(customer_data)

def data_overview(customer_data):
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(customer_data):,}")
    with col2:
        st.metric("Unique Users", f"{customer_data['user_id'].nunique():,}")
    with col3:
        st.metric("Unique Products", f"{customer_data['product_id'].nunique():,}")
    with col4:
        st.metric("Unique Brands", f"{customer_data['brand'].nunique():,}")
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(customer_data.head(10))
    
    # Basic distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(customer_data['price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Price')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Event Type Distribution")
        event_counts = customer_data['event_type'].value_counts()
        fig = px.pie(values=event_counts.values, names=event_counts.index, 
                     title="Distribution of Event Types")
        st.plotly_chart(fig, use_container_width=True)
    
    # User visit frequency
    st.subheader("User Visit Frequency Analysis")
    user_visit_counts = customer_data.groupby('user_id').size()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Visit Statistics:**")
        st.write(f"- Average visits per user: {user_visit_counts.mean():.2f}")
        st.write(f"- Median visits per user: {user_visit_counts.median():.2f}")
        st.write(f"- Max visits by a user: {user_visit_counts.max()}")
        st.write(f"- Min visits by a user: {user_visit_counts.min()}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(user_visit_counts, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.set_xlabel('Number of Visits per User')
        ax.set_ylabel('Count of Users')
        ax.set_title('Distribution of User Visit Frequency')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def visit_based_clustering(customer_data):
    st.markdown('<h2 class="section-header">👥 Visit-Based Customer Clustering</h2>', unsafe_allow_html=True)
    
    # Calculate user visit counts
    user_visit_counts = customer_data.groupby('user_id').size()
    
    # Sidebar controls
    st.sidebar.subheader("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    
    # Perform clustering
    visit_data = pd.DataFrame({
        'user_id': user_visit_counts.index,
        'visit_count': user_visit_counts.values
    })
    
    clusters, scaler, kmeans = perform_clustering(visit_data, n_clusters, ['visit_count'])
    visit_data['cluster'] = clusters
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = visit_data['cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                     labels={'x': 'Cluster', 'y': 'Number of Users'},
                     title="Users per Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Analysis")
        cluster_analysis = visit_data.groupby('cluster')['visit_count'].describe()
        st.dataframe(cluster_analysis)
    
    # Visualization
    st.subheader("Cluster Visualization")
    fig = px.scatter(visit_data, x=visit_data.index, y='visit_count', 
                     color='cluster', title="User Clusters Based on Visit Frequency",
                     labels={'x': 'User Index', 'y': 'Visit Count'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimal clusters analysis
    st.subheader("Optimal Number of Clusters Analysis")
    k_range, inertias, silhouette_scores = find_optimal_clusters(visit_data, ['visit_count'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(x=k_range, y=inertias, markers=True,
                      labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
                      title="Elbow Method")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(x=k_range, y=silhouette_scores, markers=True,
                      labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'},
                      title="Silhouette Analysis")
        st.plotly_chart(fig, use_container_width=True)

def price_based_clustering(customer_data):
    st.markdown('<h2 class="section-header">💰 Price-Based Customer Clustering</h2>', unsafe_allow_html=True)
    
    # Calculate average price per user
    user_avg_price = customer_data.groupby('user_id')['price'].mean()
    
    # Sidebar controls
    st.sidebar.subheader("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4, key="price_clusters")
    
    # Perform clustering
    price_data = pd.DataFrame({
        'user_id': user_avg_price.index,
        'avg_price': user_avg_price.values
    })
    
    clusters, scaler, kmeans = perform_clustering(price_data, n_clusters, ['avg_price'])
    price_data['cluster'] = clusters
    
    # Display basic statistics
    st.subheader("Price Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Price", f"${user_avg_price.mean():.2f}")
    with col2:
        st.metric("Median Price", f"${user_avg_price.median():.2f}")
    with col3:
        st.metric("Max Price", f"${user_avg_price.max():.2f}")
    with col4:
        st.metric("Min Price", f"${user_avg_price.min():.2f}")
    
    # Cluster analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = price_data['cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                     labels={'x': 'Cluster', 'y': 'Number of Users'},
                     title="Users per Price Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price Cluster Analysis")
        cluster_analysis = price_data.groupby('cluster')['avg_price'].describe()
        st.dataframe(cluster_analysis)
    
    # Visualizations
    st.subheader("Price Cluster Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(price_data, x=price_data.index, y='avg_price', 
                         color='cluster', title="User Clusters Based on Average Price",
                         labels={'x': 'User Index', 'y': 'Average Price ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(price_data, x='cluster', y='avg_price',
                     title="Price Distribution by Cluster",
                     labels={'cluster': 'Cluster', 'avg_price': 'Average Price ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimal clusters analysis
    st.subheader("Optimal Number of Clusters Analysis")
    k_range, inertias, silhouette_scores = find_optimal_clusters(price_data, ['avg_price'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(x=k_range, y=inertias, markers=True,
                      labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
                      title="Elbow Method for Price Clustering")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(x=k_range, y=silhouette_scores, markers=True,
                      labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'},
                      title="Silhouette Analysis for Price Clustering")
        st.plotly_chart(fig, use_container_width=True)

def multi_feature_clustering(customer_data):
    st.markdown('<h2 class="section-header">🎯 Multi-Feature Clustering</h2>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.subheader("Feature Selection")
    use_price = st.sidebar.checkbox("Include Price", True)
    use_event_type = st.sidebar.checkbox("Include Event Type", True)
    use_brand = st.sidebar.checkbox("Include Brand", False)
    
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4, key="multi_clusters")
    
    # Prepare features based on selection
    features_to_use = []
    if use_price:
        features_to_use.append('price')
    if use_event_type:
        features_to_use.append('event_type')
    if use_brand:
        features_to_use.append('brand')
    
    if not features_to_use:
        st.warning("Please select at least one feature for clustering.")
        return
    
    # Prepare data
    df_mix = customer_data[features_to_use].copy()
    
    # Handle categorical variables
    categorical_features = [f for f in features_to_use if f in ['event_type', 'brand']]
    numerical_features = [f for f in features_to_use if f not in categorical_features]
    
    if categorical_features:
        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(df_mix[categorical_features])
        encoded_columns = encoder.get_feature_names_out(categorical_features)
        
        # Combine with numerical features
        if numerical_features:
            combined_df = pd.concat([
                df_mix[numerical_features].reset_index(drop=True),
                pd.DataFrame(encoded_features, columns=encoded_columns)
            ], axis=1)
        else:
            combined_df = pd.DataFrame(encoded_features, columns=encoded_columns)
    else:
        combined_df = df_mix[numerical_features]
    
    # Scale features if needed
    if numerical_features:
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_df)
    else:
        combined_scaled = combined_df.values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(combined_scaled)
    
    df_mix['cluster'] = clusters
    
    # Display results
    st.subheader(f"Clustering Results using: {', '.join(features_to_use)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                     labels={'x': 'Cluster', 'y': 'Number of Records'},
                     title="Records per Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Composition")
        if 'price' in features_to_use and 'event_type' in features_to_use:
            composition = df_mix.groupby('cluster').agg({
                'price': 'mean' if 'price' in features_to_use else lambda x: 'N/A',
                'event_type': lambda x: x.value_counts().index[0] if 'event_type' in features_to_use else 'N/A'
            })
            st.dataframe(composition)
    
    # Elbow method for optimal clusters
    if st.button("Find Optimal Number of Clusters"):
        st.subheader("Optimal Clusters Analysis")
        
        k_range = range(2, 11)
        inertias = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(combined_scaled)
            inertias.append(km.inertia_)
        
        fig = px.line(x=k_range, y=inertias, markers=True,
                      labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
                      title=f"Elbow Method for {', '.join(features_to_use)}")
        st.plotly_chart(fig, use_container_width=True)

def advanced_behavior_analysis(customer_data):
    st.markdown('<h2 class="section-header">🧠 Advanced User Behavior Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate comprehensive user features
    user_features = calculate_user_metrics(customer_data)
    
    # Sidebar controls
    st.sidebar.subheader("Advanced Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 15, 10, key="advanced_clusters")
    
    feature_options = ['visit_frequency', 'view_to_action_ratio', 'brand_interactions', 'avg_price']
    selected_features = st.sidebar.multiselect(
        "Select Features for Clustering",
        feature_options,
        default=['visit_frequency', 'view_to_action_ratio', 'brand_interactions']
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for clustering.")
        return
    
    # Display feature distributions
    st.subheader("Feature Distributions")
    
    n_features = len(selected_features)
    cols = st.columns(min(n_features, 3))
    
    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(user_features[feature], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Number of Users')
            ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Perform clustering
    clustering_features = user_features[['user_id'] + selected_features].copy()
    
    # Prepare data for clustering
    X = clustering_features[selected_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    clustering_features['cluster'] = clusters
    
    # Display clustering results
    st.subheader("Clustering Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = clustering_features['cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                     labels={'x': 'Cluster', 'y': 'Number of Users'},
                     title="Users per Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Correlations")
        corr_matrix = clustering_features[selected_features].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed cluster analysis
    st.subheader("Detailed Cluster Analysis")
    
    cluster_analysis = clustering_features.groupby('cluster')[selected_features].agg(['mean', 'std']).round(3)
    st.dataframe(cluster_analysis)
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    
    for cluster_id in sorted(clustering_features['cluster'].unique()):
        cluster_data = clustering_features[clustering_features['cluster'] == cluster_id]
        size = len(cluster_data)
        
        # Calculate averages for profiling
        profile_metrics = {}
        for feature in selected_features:
            profile_metrics[feature] = cluster_data[feature].mean()
        
        # Create profile description
        profile_parts = []
        
        if 'visit_frequency' in selected_features:
            avg_visits = profile_metrics['visit_frequency']
            visit_level = "High" if avg_visits > clustering_features['visit_frequency'].median() else "Low"
            profile_parts.append(f"{visit_level} Activity")
        
        if 'view_to_action_ratio' in selected_features:
            avg_ratio = profile_metrics['view_to_action_ratio']
            engagement = "High Conversion" if avg_ratio > clustering_features['view_to_action_ratio'].median() else "Low Conversion"
            profile_parts.append(engagement)
        
        if 'brand_interactions' in selected_features:
            avg_brands = profile_metrics['brand_interactions']
            brand_behavior = "Multi-Brand" if avg_brands > clustering_features['brand_interactions'].median() else "Single-Brand"
            profile_parts.append(brand_behavior)
        
        if 'avg_price' in selected_features:
            avg_price_val = profile_metrics['avg_price']
            price_level = "High Value" if avg_price_val > clustering_features['avg_price'].median() else "Low Value"
            profile_parts.append(price_level)
        
        profile = ", ".join(profile_parts)
        
        # Display cluster profile
        st.markdown(f"""
        <div class="cluster-profile">
            <h4>Cluster {cluster_id} ({size:,} users)</h4>
            <p><strong>Profile:</strong> {profile}</p>
            <ul>
        """, unsafe_allow_html=True)
        
        for feature in selected_features:
            value = profile_metrics[feature]
            if feature == 'avg_price':
                st.markdown(f"<li>{feature.replace('_', ' ').title()}: ${value:.2f}</li>", unsafe_allow_html=True)
            else:
                st.markdown(f"<li>{feature.replace('_', ' ').title()}: {value:.3f}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Optimal clusters analysis
    if st.button("Find Optimal Number of Clusters", key="advanced_optimal"):
        st.subheader("Optimal Clusters Analysis")
        k_range, inertias, silhouette_scores = find_optimal_clusters(clustering_features, selected_features, max_k=15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(x=k_range, y=inertias, markers=True,
                          labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
                          title="Elbow Method")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(x=k_range, y=silhouette_scores, markers=True,
                          labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'},
                          title="Silhouette Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        optimal_k_silhouette = k_range[silhouette_scores.index(max(silhouette_scores))]
        st.success(f"Recommended number of clusters based on Silhouette Score: **{optimal_k_silhouette}**")

if __name__ == "__main__":
    main()
