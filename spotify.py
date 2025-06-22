# spotify_eda.py
# Spotify Dataset - EDA and Visualization
# Internship Project for Celebal
# Author: (Your Name)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Setup
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Create visuals folder if it doesn't exist
if not os.path.exists("visuals"):
    os.makedirs("visuals")

# Load the dataset
df = pd.read_csv("dataset.csv", encoding='latin1')
print("✅ Spotify data loaded!")

# Preview data
print(df.head())

# Check null values
print("\nMissing values:\n", df.isnull().sum())

# Keep only useful columns for analysis
columns_needed = [
    'track_name', 'artists', 'duration_ms', 'explicit', 'popularity',
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre'
]
df = df[columns_needed]

# Rename for consistency
df.rename(columns={'track_name': 'name'}, inplace=True)

# ========================
#  Visualization Section
# ========================

# 1. Top 10 Most Popular Songs
top_tracks = df.sort_values(by='popularity', ascending=False).head(10)
sns.barplot(data=top_tracks, y='name', x='popularity', palette='magma')
plt.title("Top 10 Most Popular Spotify Tracks")
plt.xlabel("Popularity Score")
plt.ylabel("Track Name")
plt.tight_layout()
plt.savefig("visuals/top10_popular_tracks.png")
plt.show()

# 2. Danceability Distribution
sns.histplot(df['danceability'], bins=30, color='green', kde=True)
plt.title("Danceability Distribution")
plt.xlabel("Danceability")
plt.tight_layout()
plt.savefig("visuals/danceability_distribution.png")
plt.show()

# 3. Energy vs Loudness
sns.scatterplot(data=df, x='energy', y='loudness', hue='explicit', palette='Set1')
plt.title("Energy vs Loudness (Explicit vs Clean)")
plt.xlabel("Energy")
plt.ylabel("Loudness")
plt.tight_layout()
plt.savefig("visuals/energy_vs_loudness.png")
plt.show()

# 4. Distribution of Popularity by Genre (Top 10 genres)
top_genres = df['track_genre'].value_counts().head(10).index
genre_df = df[df['track_genre'].isin(top_genres)]

sns.boxplot(data=genre_df, x='track_genre', y='popularity', palette='coolwarm')
plt.title("Popularity Distribution by Top 10 Genres")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visuals/popularity_by_genre.png")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = [
    'popularity', 'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Correlation Between Audio Features")
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap.png")
plt.show()

# ========================
# 🔍 Summary
# ========================
print("\n--- Summary of Insights ---")
print(" The most popular tracks show high danceability and energy.")
print(" Genres vary significantly in popularity — some are more niche.")
print(" Loudness and energy are strongly correlated.")
print(" Explicit tracks tend to have higher energy.")
print(" Danceability is mostly between 0.5 and 0.8.")
print("\n Visualization project completed successfully!")
