from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

# Load the new trained model (with 12 features)
import os

MODEL_PATH = "fairwork_ai_model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        model = None
else:
    print("❌ Model file not found! Ensure fairwork_ai_model.pkl exists.")
    model = None


# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "FairWork AI is running!"}

@app.post("/predict/")
def predict_burnout(
    gender: str, age: int, department: str, experience_level: str, 
    tasks_assigned: int, task_difficulty: int, hours_worked: int, 
    deadline_days: int, stress_level: int, sick_leaves: int, 
    self_reported_fatigue: int, productivity_score: float
):
    # Ensure model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check model file.")

    # Categorical mappings
    gender_map = {"Male": 0, "Female": 1, "Non-Binary": 2}
    department_map = {"Engineering": 0, "HR": 1, "Marketing": 2, "Sales": 3, "Finance": 4}
    experience_map = {"Junior": 0, "Mid": 1, "Senior": 2}

    # Convert categorical inputs
    gender_encoded = gender_map.get(gender, 0)
    department_encoded = department_map.get(department, 0)
    experience_encoded = experience_map.get(experience_level, 0)

    # Prepare input data (ONLY 12 FEATURES)
    input_data = np.array([
        gender_encoded, age, department_encoded, experience_encoded,
        tasks_assigned, task_difficulty, hours_worked, deadline_days, 
        stress_level, sick_leaves, self_reported_fatigue, productivity_score
    ]).reshape(1, -1)

    # Predict burnout risk
    try:
        prediction = model.predict(input_data)
        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        return {"Burnout Risk Prediction": risk_labels.get(prediction[0], "Unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
