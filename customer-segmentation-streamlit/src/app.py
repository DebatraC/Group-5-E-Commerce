import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="📊",
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
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_and_preprocess_data():
    """Load and preprocess the customer data"""
    try:
        # Try to load the data from the workspace
        data_path = '../mytestdata.parquet'
        data = pd.read_parquet(data_path)
        return data
    except:
        # If file not found, return None
        return None

def fill_missing_brand(group):
    """Fill missing brand values using forward and backward fill"""
    if group['brand'].notna().any():
        group['brand'] = group['brand'].fillna(method='ffill').fillna(method='bfill')
    return group

def preprocess_data(data):
    """Preprocess the data with brand filling"""
    # Fill missing brand values
    data = data.groupby('product_id').apply(fill_missing_brand).reset_index(drop=True)
    return data

def perform_visit_clustering(data, n_clusters=4):
    """Perform clustering based on user visit frequency"""
    # Group by user_id and count visits
    user_visit_counts = data.groupby('user_id').size()
    
    # Scale the data
    scaler = StandardScaler()
    visit_counts_scaled = scaler.fit_transform(user_visit_counts.values.reshape(-1, 1))
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(visit_counts_scaled)
    
    # Create results dataframe
    user_clusters = pd.DataFrame({
        'user_id': user_visit_counts.index,
        'visit_count': user_visit_counts.values,
        'cluster': clusters
    })
    
    return user_clusters, kmeans

def perform_price_clustering(data, n_clusters=4):
    """Perform clustering based on average price per user"""
    # Calculate average price per user
    user_avg_price = data.groupby('user_id')['price'].mean()
    
    # Scale the data
    scaler = StandardScaler()
    avg_price_scaled = scaler.fit_transform(user_avg_price.values.reshape(-1, 1))
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(avg_price_scaled)
    
    # Create results dataframe
    user_price_clusters = pd.DataFrame({
        'user_id': user_avg_price.index,
        'avg_price': user_avg_price.values,
        'cluster': clusters
    })
    
    return user_price_clusters, kmeans

def perform_price_event_clustering(data, n_clusters=4):
    """Perform clustering based on price and event type"""
    df_mix = data[['price', 'event_type']].copy()
    
    # One-hot encode event_type
    encoder = OneHotEncoder(sparse_output=False)
    event_encoded = encoder.fit_transform(df_mix[['event_type']])
    event_columns = encoder.get_feature_names_out(['event_type'])
    
    # Combine price with encoded event_type
    combined_df = pd.concat([
        df_mix[['price']].reset_index(drop=True),
        pd.DataFrame(event_encoded, columns=event_columns)
    ], axis=1)
    
    # Scale all features
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_df)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(combined_scaled)
    
    df_mix['cluster'] = clusters
    
    return df_mix, kmeans, combined_scaled

def perform_brand_event_clustering(data, n_clusters=8):
    """Perform clustering based on brand and event type"""
    df_mix = data[['brand', 'event_type']].copy()
    
    # One-hot encode both brand and event_type
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df_mix[['brand', 'event_type']])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(encoded_features)
    
    df_mix['cluster'] = clusters
    
    return df_mix, kmeans, encoded_features

def perform_advanced_clustering(data, n_clusters=10):
    """Perform advanced clustering using multiple user behavior features"""
    # Calculate user-level features
    user_metrics = data.groupby('user_id').agg({
        'event_type': ['count'],
        'brand': 'nunique'
    }).reset_index()
    
    # Flatten column names
    user_metrics.columns = ['user_id', 'visit_frequency', 'brand_interactions']
    
    # Calculate event type counts per user
    event_counts = data.groupby(['user_id', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Merge with user metrics
    user_features = user_metrics.merge(event_counts, on='user_id', how='left')
    
    # Calculate view-to-action ratio
    user_features['view_to_action_ratio'] = np.where(
        user_features['view'] > 0,
        (user_features.get('cart', 0) + user_features.get('purchase', 0)) / user_features['view'],
        0
    )
    
    # Prepare clustering features
    clustering_features = user_features[['user_id', 'visit_frequency', 'view_to_action_ratio', 'brand_interactions']].copy()
    
    # Select features for clustering
    X = clustering_features[['visit_frequency', 'view_to_action_ratio', 'brand_interactions']].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    clustering_features['cluster'] = clusters
    
    return clustering_features, kmeans, X_scaled

def calculate_elbow_method(X_scaled, max_k=15):
    """Calculate inertia values for elbow method"""
    k_range = range(2, max_k + 1)
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    return k_range, inertias

def plot_elbow_curve(k_range, inertias):
    """Plot elbow curve using plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=inertias,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Elbow Method for Optimal Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_cluster_distribution(clusters_df, cluster_col='cluster', title="Cluster Distribution"):
    """Plot cluster distribution using plotly"""
    cluster_counts = clusters_df[cluster_col].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(x=cluster_counts.index, y=cluster_counts.values,
               marker_color='#1f77b4')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Cluster',
        yaxis_title='Number of Users/Records',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_feature_histograms(data, features):
    """Plot histograms for multiple features"""
    fig = make_subplots(
        rows=1, cols=len(features),
        subplot_titles=features
    )
    
    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(x=data[feature], name=feature, showlegend=False),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title='Distribution of Features',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_cluster_analysis(clusters_df, features, cluster_col='cluster'):
    """Plot cluster analysis for multiple features"""
    n_features = len(features)
    fig = make_subplots(
        rows=1, cols=n_features,
        subplot_titles=features
    )
    
    for i, feature in enumerate(features):
        for cluster in sorted(clusters_df[cluster_col].unique()):
            cluster_data = clusters_df[clusters_df[cluster_col] == cluster]
            fig.add_trace(
                go.Box(y=cluster_data[feature], name=f'Cluster {cluster}', 
                      showlegend=(i == 0)),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title='Cluster Analysis by Features',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">📊 Customer Segmentation Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Data Overview", "Visit-Based Clustering", "Price-Based Clustering", 
         "Price-Event Clustering", "Brand-Event Clustering", "Advanced Clustering"]
    )
    
    # Load data
    data = load_and_preprocess_data()
    
    if data is None:
        st.error("Could not load data. Please ensure the data file exists in the correct location.")
        st.info("Expected location: ../mytestdata.parquet")
        
        # Allow file upload as fallback
        st.markdown("### Upload Data File")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'parquet'])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_parquet(uploaded_file)
        else:
            st.stop()
    
    # Preprocess data
    data = preprocess_data(data)
    
    if page == "Data Overview":
        st.markdown('<h2 class="section-header">📈 Data Overview</h2>', unsafe_allow_html=True)
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Unique Users", f"{data['user_id'].nunique():,}")
        with col3:
            st.metric("Unique Products", f"{data['product_id'].nunique():,}")
        with col4:
            st.metric("Unique Brands", f"{data['brand'].nunique():,}")
        
        # Show data sample
        st.subheader("Data Sample")
        st.dataframe(data.head(10))
        
        # Show data statistics
        st.subheader("Data Statistics")
        st.dataframe(data.describe())
        
        # Event type distribution
        st.subheader("Event Type Distribution")
        event_counts = data['event_type'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=event_counts.index, values=event_counts.values)])
        fig.update_layout(title="Distribution of Event Types")
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution
        st.subheader("Price Distribution")
        fig = go.Figure(data=[go.Histogram(x=data['price'], nbinsx=50)])
        fig.update_layout(title="Distribution of Prices", xaxis_title="Price", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Visit-Based Clustering":
        st.markdown('<h2 class="section-header">👥 Visit-Based Clustering</h2>', unsafe_allow_html=True)
        
        # Clustering parameters
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
        
        if st.button("Run Visit-Based Clustering"):
            with st.spinner("Performing visit-based clustering..."):
                user_clusters, kmeans = perform_visit_clustering(data, n_clusters)
                
                # Display results
                st.success("Clustering completed!")
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                fig = plot_cluster_distribution(user_clusters, 'cluster', "Visit-Based Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster analysis
                st.subheader("Cluster Analysis")
                cluster_stats = user_clusters.groupby('cluster')['visit_count'].describe()
                st.dataframe(cluster_stats)
                
                # Visit count distribution by cluster
                st.subheader("Visit Count Distribution by Cluster")
                fig = plot_cluster_analysis(user_clusters, ['visit_count'], 'cluster')
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Price-Based Clustering":
        st.markdown('<h2 class="section-header">💰 Price-Based Clustering</h2>', unsafe_allow_html=True)
        
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
        show_elbow = st.sidebar.checkbox("Show Elbow Method")
        
        if st.button("Run Price-Based Clustering"):
            with st.spinner("Performing price-based clustering..."):
                user_price_clusters, kmeans = perform_price_clustering(data, n_clusters)
                
                st.success("Clustering completed!")
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                fig = plot_cluster_distribution(user_price_clusters, 'cluster', "Price-Based Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster analysis
                st.subheader("Cluster Analysis")
                cluster_stats = user_price_clusters.groupby('cluster')['avg_price'].describe()
                st.dataframe(cluster_stats)
                
                # Price distribution by cluster
                st.subheader("Price Distribution by Cluster")
                fig = plot_cluster_analysis(user_price_clusters, ['avg_price'], 'cluster')
                st.plotly_chart(fig, use_container_width=True)
                
                if show_elbow:
                    st.subheader("Elbow Method Analysis")
                    user_avg_price = data.groupby('user_id')['price'].mean()
                    scaler = StandardScaler()
                    avg_price_scaled = scaler.fit_transform(user_avg_price.values.reshape(-1, 1))
                    
                    k_range, inertias = calculate_elbow_method(avg_price_scaled)
                    fig = plot_elbow_curve(k_range, inertias)
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Price-Event Clustering":
        st.markdown('<h2 class="section-header">🏷️ Price-Event Clustering</h2>', unsafe_allow_html=True)
        
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
        
        if st.button("Run Price-Event Clustering"):
            with st.spinner("Performing price-event clustering..."):
                df_mix, kmeans, combined_scaled = perform_price_event_clustering(data, n_clusters)
                
                st.success("Clustering completed!")
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                fig = plot_cluster_distribution(df_mix, 'cluster', "Price-Event Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster composition
                st.subheader("Cluster Composition")
                cluster_composition = df_mix.groupby('cluster').agg({
                    'price': 'mean',
                    'event_type': lambda x: x.value_counts().index[0]
                })
                st.dataframe(cluster_composition)
                
                # Elbow method
                st.subheader("Elbow Method Analysis")
                k_range, inertias = calculate_elbow_method(combined_scaled)
                fig = plot_elbow_curve(k_range, inertias)
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Brand-Event Clustering":
        st.markdown('<h2 class="section-header">🏢 Brand-Event Clustering</h2>', unsafe_allow_html=True)
        
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 15, 8)
        
        if st.button("Run Brand-Event Clustering"):
            with st.spinner("Performing brand-event clustering..."):
                df_mix, kmeans, encoded_features = perform_brand_event_clustering(data, n_clusters)
                
                st.success("Clustering completed!")
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                fig = plot_cluster_distribution(df_mix, 'cluster', "Brand-Event Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster composition
                st.subheader("Cluster Composition")
                cluster_composition = df_mix.groupby('cluster').agg({
                    'brand': lambda x: x.value_counts().index[0],
                    'event_type': lambda x: x.value_counts().index[0]
                })
                st.dataframe(cluster_composition)
                
                # Elbow method
                st.subheader("Elbow Method Analysis")
                k_range, inertias = calculate_elbow_method(encoded_features)
                fig = plot_elbow_curve(k_range, inertias)
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Advanced Clustering":
        st.markdown('<h2 class="section-header">🚀 Advanced User Behavior Clustering</h2>', unsafe_allow_html=True)
        
        st.info("This clustering approach uses visit frequency, view-to-action ratio, and brand interactions to create comprehensive user segments.")
        
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 10)
        show_elbow = st.sidebar.checkbox("Show Elbow Method", value=True)
        
        if st.button("Run Advanced Clustering"):
            with st.spinner("Performing advanced clustering analysis..."):
                clustering_features, kmeans, X_scaled = perform_advanced_clustering(data, n_clusters)
                
                st.success("Advanced clustering completed!")
                
                # Display feature distributions
                st.subheader("Feature Distributions")
                features = ['visit_frequency', 'view_to_action_ratio', 'brand_interactions']
                fig = plot_feature_histograms(clustering_features, features)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature correlation
                st.subheader("Feature Correlations")
                correlation_matrix = clustering_features[features].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    text=correlation_matrix.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 12}
                ))
                fig.update_layout(title="Feature Correlation Matrix", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                if show_elbow:
                    st.subheader("Elbow Method Analysis")
                    k_range, inertias = calculate_elbow_method(X_scaled, 20)
                    fig = plot_elbow_curve(k_range, inertias)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                fig = plot_cluster_distribution(clustering_features, 'cluster', "Advanced Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster analysis
                st.subheader("Cluster Analysis by Features")
                fig = plot_cluster_analysis(clustering_features, features, 'cluster')
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed cluster profiles
                st.subheader("Detailed Cluster Profiles")
                
                cluster_analysis = clustering_features.groupby('cluster').agg({
                    'visit_frequency': ['mean', 'std', 'min', 'max'],
                    'view_to_action_ratio': ['mean', 'std', 'min', 'max'],
                    'brand_interactions': ['mean', 'std', 'min', 'max']
                }).round(3)
                
                st.dataframe(cluster_analysis)
                
                # Business insights
                st.subheader("🎯 Business Insights")
                
                median_visits = clustering_features['visit_frequency'].median()
                median_ratio = clustering_features['view_to_action_ratio'].median()
                median_brands = clustering_features['brand_interactions'].median()
                
                for cluster_id in sorted(clustering_features['cluster'].unique()):
                    cluster_data = clustering_features[clustering_features['cluster'] == cluster_id]
                    
                    avg_visits = cluster_data['visit_frequency'].mean()
                    avg_ratio = cluster_data['view_to_action_ratio'].mean()
                    avg_brands = cluster_data['brand_interactions'].mean()
                    size = len(cluster_data)
                    
                    # Create profile name
                    visit_level = "High" if avg_visits > median_visits else "Low"
                    engagement = "High Conversion" if avg_ratio > median_ratio else "Low Conversion"
                    brand_behavior = "Multi-Brand" if avg_brands > median_brands else "Single-Brand"
                    
                    profile = f"{visit_level} Activity, {engagement}, {brand_behavior}"
                    
                    with st.expander(f"🎯 Cluster {cluster_id}: {profile} ({size:,} users)"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg Visits", f"{avg_visits:.1f}")
                        with col2:
                            st.metric("Avg Conversion Ratio", f"{avg_ratio:.3f}")
                        with col3:
                            st.metric("Avg Brand Interactions", f"{avg_brands:.1f}")
                        
                        # Business recommendations
                        st.markdown("**💡 Business Recommendations:**")
                        
                        if visit_level == "High" and engagement == "High Conversion":
                            st.markdown("- 🌟 **VIP Customers**: Focus on retention and premium offerings")
                            st.markdown("- 📧 Personalized email campaigns with exclusive deals")
                            st.markdown("- 🎁 Loyalty program benefits and early access")
                        elif visit_level == "High" and engagement == "Low Conversion":
                            st.markdown("- 🔍 **High Browsers**: Need conversion optimization")
                            st.markdown("- 💰 Targeted discounts and limited-time offers")
                            st.markdown("- 📱 Retargeting campaigns and cart abandonment emails")
                        elif visit_level == "Low" and engagement == "High Conversion":
                            st.markdown("- 🎯 **Efficient Buyers**: Focus on acquisition")
                            st.markdown("- 📈 Increase visit frequency through engagement")
                            st.markdown("- 🔔 Push notifications for new products")
                        else:
                            st.markdown("- 📢 **Need Attention**: Requires activation campaigns")
                            st.markdown("- 🎁 Welcome series and onboarding improvements")
                            st.markdown("- 💌 Re-engagement campaigns with incentives")

if __name__ == "__main__":
    main()