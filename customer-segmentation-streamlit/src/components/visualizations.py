import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_visit_distribution(user_clusters):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=user_clusters, x='cluster', palette='viridis')
    plt.title('Cluster Distribution of Users')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_average_price_distribution(user_price_clusters):
    plt.figure(figsize=(10, 6))
    sns.histplot(user_price_clusters['avg_price'], bins=30, kde=True, color='blue', alpha=0.6)
    plt.title('Distribution of Average Price per User')
    plt.xlabel('Average Price ($)')
    plt.ylabel('Number of Users')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_user_clusters(user_clusters):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=user_clusters, x='visit_count', y='cluster', hue='cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('User Clusters Based on Visit Count')
    plt.xlabel('Visit Count')
    plt.ylabel('Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()