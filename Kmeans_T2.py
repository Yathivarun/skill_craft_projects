import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("C:/Users/pdx/OneDrive/Desktop/mini_project/z_internship/datasets/Mall_Customers.csv")
df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)

# Feature selection
features = df[['Income', 'Score']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow Method for optimal k
inertias = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Apply KMeans with k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Cluster visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Income', y='Score', hue='Cluster', palette='Set1')
plt.title("Customer Segments based on Income and Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Centroid info
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centroids:")
for i, center in enumerate(centroids):
    print(f"Cluster {i}: Income = {center[0]:.2f}, Score = {center[1]:.2f}")

# Cluster Descriptions
descriptions = {
    0: "Average Income, Average Spending",
    1: "High Income, High Spending",
    2: "Low Income, High Spending",
    3: "High Income, Low Spending",
    4: "Low Income, Low Spending"
}

# Optional: Pie chart for cluster distribution
df['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Customer Cluster Distribution")
plt.ylabel("")
plt.show()

# ----------- USER PREDICTION SECTION -----------

print("\n--- Predict Your Customer Segment ---")
try:
    user_income = float(input("Enter your Annual Income (in $1000s) [15–140]: "))
    user_income = max(15, min(140, user_income))  # Clamp to dataset range

    user_score = float(input("Enter your Spending Score (1–100): "))
    user_score = max(1, min(100, user_score))  # Clamp to valid range

    # Convert to DataFrame to avoid warning
    user_df = pd.DataFrame([[user_income, user_score]], columns=['Income', 'Score'])
    user_scaled = scaler.transform(user_df)
    user_cluster = kmeans.predict(user_scaled)[0]

    print(f"\nBased on your profile:")
    print(f"  → You belong to Cluster {user_cluster}: {descriptions[user_cluster]}")

except ValueError:
    print("❌ Invalid input. Please enter numeric values only.")
