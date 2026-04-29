from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load dataset and train model
iris = load_iris()
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)

# Class names
class_names = ["setosa", "versicolor", "virginica"]

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# Prediction endpoint
@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):
    features = np.array([[sl, sw, pl, pw]])
    pred = int(model.predict(features)[0])
    
    return {
        "prediction": pred,
        "class_name": class_names[pred]
    }
