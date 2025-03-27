from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load the trained K-Means model
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Initialize FastAPI app
app = FastAPI(title="Mall Customer Segmentation API")

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (Change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Mall Customer Segmentation API"}

# Endpoint to predict customer segment
@app.post("/predict")
def predict_cluster(age: int, income: int, spending_score: int):
    # Prepare input data
    input_data = np.array([[age, income, spending_score]])

    # Predict cluster
    cluster = kmeans.predict(input_data)[0]
    
    return {
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score": spending_score,
        "Predicted Cluster": int(cluster)
    }

# Run the app with: uvicorn app:app --reload



# @app.post("/predict")
# def predict_cluster(age: int, income: int, spending_score: int):
#     input_data = np.array([[age, income, spending_score]])
#     input_scaled = kmeans.transform(input_data)  # Apply the same scaling
#     cluster = kmeans.predict(input_scaled)[0]
    
#     return {
#         "Age": age,
#         "Annual Income (k$)": income,
#         "Spending Score": spending_score,
#         "Predicted Cluster": int(cluster)
#     }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



