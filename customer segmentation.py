import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# -------------------------
# 1. Load and preprocess data
# -------------------------
df = pd.read_excel("Online Retail(AutoRecovered).xlsx")

df = df.dropna(subset=["CustomerID"])          # remove missing customers
df = df[df["Quantity"] > 0]                    # keep only positive quantities

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "Quantity": lambda q: np.sum(q * df.loc[q.index, "UnitPrice"])
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# -------------------------
# 2. Normalize RFM
# -------------------------
rfm_norm = (rfm[["Recency","Frequency","Monetary"]] - rfm[["Recency","Frequency","Monetary"]].min()) / \
           (rfm[["Recency","Frequency","Monetary"]].max() - rfm[["Recency","Frequency","Monetary"]].min())

rfm_norm["CustomerID"] = rfm["CustomerID"]

# -------------------------
# 3. Helper functions for K-means
# -------------------------
def distance(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def init_centroids(data, k, seed=42):
    random.seed(seed)
    return random.sample(data, k)

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        dists = [distance(point, c) for c in centroids]
        clusters[dists.index(min(dists))].append(point)
    return clusters

def compute_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(col)/len(col) for col in zip(*cluster)]
            centroids.append(centroid)
    return centroids

def kmeans(data, k=4, max_iters=100, seed=42):
    centroids = init_centroids(data, k, seed)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = compute_centroids(clusters)
        # Use np.allclose for floating-point comparison
        if all(np.allclose(nc, c) for nc, c in zip(new_centroids, centroids)):
            break
        centroids = new_centroids
    return clusters, centroids

def compute_sse(clusters, centroids):
    sse = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            sse += distance(point, centroids[i]) ** 2
    return sse

def assign_cluster_label(row, centroids):
    point = [row["Recency"], row["Frequency"], row["Monetary"]]
    dists = [distance(point, c) for c in centroids]
    return np.argmin(dists)

# -------------------------
# 4. Prepare data for clustering
# -------------------------
data = rfm_norm[["Recency","Frequency","Monetary"]].values.tolist()

# -------------------------
# 5. Determine optimal K using Elbow
# -------------------------
Ks = range(1, 11)
sses = []

for k in Ks:
    cl, ce = kmeans(data, k)
    sses.append(compute_sse(cl, ce))

plt.plot(Ks, sses, marker="o")
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.grid()
plt.show()

# -------------------------
# 6. Run K-means with multiple initializations to get best clustering
# -------------------------
k = 4  # choose based on elbow
best_sse = float("inf")
best_clusters = None
best_centroids = None

for _ in range(10):  # 10 random initializations
    cl, ce = kmeans(data, k, seed=random.randint(0, 10000))
    sse = compute_sse(cl, ce)
    if sse < best_sse:
        best_sse = sse
        best_clusters = cl
        best_centroids = ce

clusters, centroids = best_clusters, best_centroids

# -------------------------
# 7. Assign cluster labels robustly
# -------------------------
rfm_norm["Cluster"] = rfm_norm.apply(lambda row: assign_cluster_label(row, centroids), axis=1)

# -------------------------
# 8. Visualize cluster centers
# -------------------------
plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(centroids, columns=["Recency","Frequency","Monetary"]),
            annot=True, cmap="coolwarm")
plt.title("Cluster Center Heatmap")
plt.show()

# -------------------------
# 9. Boxplots of clusters
# -------------------------
plt.figure(figsize=(14,4))
for i, col in enumerate(["Recency","Frequency","Monetary"]):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="Cluster", y=col, data=rfm_norm)
    plt.title(f"{col} by Cluster")
plt.show()

# -------------------------
# 10. Cluster summary
# -------------------------
for i in range(k):
    cluster_data = rfm_norm[rfm_norm["Cluster"] == i]
    print(f"\nCluster {i} Size:", len(cluster_data))
    print(cluster_data[["Recency","Frequency","Monetary"]].mean())
