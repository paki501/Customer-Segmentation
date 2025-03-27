# Customer-Segmentation using Clustering
This project segments customers based on age, income, and spending behavior using unsupervised learning (K-Means Clustering). The goal is to identify distinct customer groups to help businesses optimize their marketing strategies.
# Getting Started
Follow these steps to set up and run the project on your local machine for development and testing.
## Required Libraries
+ FastAPI – Backend API

+ Uvicorn – ASGI server for FastAPI

+ Scikit-learn – Machine learning models

+ Pandas – Data processing

+ Matplotlib & Seaborn – Data visualization

+ NumPy – Numerical computations

+ Pickle – Model serialization
# Dataset Used
The dataset consists of mall customer information, including age, income, and spending score.

Columns:

+ CustomerID → Unique customer identifier

+ Gender → Male/Female

+ Age → Customer’s age

+ Annual Income (k$) → Income per year

+ Spending Score (1-100) → Spending behavior
## Data Preprocessing Steps:
1 Feature Selection – Used Age, Annual Income, and Spending Score

2 Data Scaling – Applied StandardScaler to normalize values

3 Finding Optimal Clusters (k) – Used Elbow Method
# Model Training & Challenges
### Model Used
K-Means Clustering – Unsupervised learning model for customer segmentation
## Training Process
### Data Preprocessing:

Handled missing values (if any)

Scaled numerical features using StandardScaler

### Applying K-Means Clustering:

Used Elbow Method to determine the optimal number of clusters (k)

Trained K-Means Model on processed data

### Model Saving:

Saved the trained K-Means model (kmeans_model.pkl)

Saved the Scaler (scaler.pkl) to apply transformations to new inputs
## Challenges Faced
### Choosing the Right k (Number of Clusters):

Used Elbow Method to find the best k (optimal cluster count).

Experimented with k=3, k=5, k=6 to get meaningful segments.
###  Scaling Data for Better Cluster Formation:

Without scaling, K-Means gave biased clusters due to large income values.

Applied StandardScaler for normalization.
### Imbalanced Cluster Distribution:

Some initial results assigned most customers to one cluster.

Improved by optimizing k value and checking cluster sizes.
# Model Performance & Visualizations
To evaluate segmentation quality, we used multiple visualizations:

+ Elbow Curve – Helps select the best k for clustering
+ Pairplots – Shows relationships between age, income, and spending
+ 3D Scatter Plot – Visualizes clusters in 3D space
+ Cluster Distribution Plot – Checks the balance of formed clusters








