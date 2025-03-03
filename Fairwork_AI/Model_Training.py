#Importing Dependencies
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of employees
num_employees = 500

# Generate Synthetic Employee Data
genders = ['Male', 'Female', 'Non-Binary']
departments = ['Engineering', 'HR', 'Marketing', 'Sales', 'Finance']
experience_levels = ['Junior', 'Mid', 'Senior']

# Generate Employee Demographics Data
employee_demographics = {
    'Employee_ID': range(1,num_employees+1),
    'Gender': np.random.choice(genders, num_employees, p=[0.50, 0.45, 0.05]),
    'Age': np.random.randint(22, 60, num_employees),
    'Department': np.random.choice(departments, num_employees),
    'Experience_Level': np.random.choice(experience_levels, num_employees, p=[0.40, 0.40, 0.20])
}
demographics_df = pd.DataFrame(employee_demographics)

# Generate Work Data
work_data = {
    'Employee_ID': demographics_df['Employee_ID'],
    'Tasks_Assigned': np.random.randint(5, 20, num_employees),
    'Task_Difficulty': np.random.randint(1, 10, num_employees),
    'Hours_Worked': np.random.randint(30, 60, num_employees),
    'Deadline_Days': np.random.randint(1, 30, num_employees)  
}
work_df = pd.DataFrame(work_data)

# Generate Stress & Burnout Data
stress_burnout_data = {
    'Employee_ID': demographics_df['Employee_ID'],
    'Stress_Level': np.random.randint(1, 10, num_employees),  
    'Burnout_Risk': np.random.choice(['Low', 'Medium', 'High'], num_employees, p=[0.60, 0.30, 0.10]),
    'Sick_Leaves': np.random.randint(0, 10, num_employees),  
    'Self_Reported_Fatigue': np.random.randint(1, 10, num_employees), 
    'Productivity_Score': np.random.uniform(0.5, 1.5, num_employees)  
}
stress_df = pd.DataFrame(stress_burnout_data)

# Save datasets
demographics_df.to_csv('employee_demographics.csv', index=False)
work_df.to_csv('work_data.csv', index=False)
stress_df.to_csv('stress_burnout.csv', index=False)

# Display first few rows
print("Employee Demographics:")
print(demographics_df.head())
print("\nWork Data:")
print(work_df.head())
print("\nStress & Burnout Data:")
print(stress_df.head())

# Merge the datasets using Employee_ID as the key
full_df = demographics_df.merge(work_df, on='Employee_ID').merge(stress_df, on='Employee_ID')

# Check for missing values
if full_df.isnull().sum().sum() > 0:
    print("⚠️ Warning: The dataset contains missing values.")
else:
    print("✅ No missing values found in the dataset.")

# Check for duplicate values
if full_df.duplicated().sum() > 0:
    print(f"⚠️ Warning: The dataset contains {full_df.duplicated().sum()} duplicate rows.")
else:
    print("✅ No duplicate values found in the dataset.")

print(full_df.describe())  # Summary statistics
print(full_df['Burnout_Risk'].value_counts())  # Count of each risk level

# Encode categorical variables
encoder = LabelEncoder()
full_df['Gender'] = encoder.fit_transform(full_df['Gender'])
full_df['Department'] = encoder.fit_transform(full_df['Department'])
full_df['Experience_Level'] = encoder.fit_transform(full_df['Experience_Level'])
full_df['Burnout_Risk'] = encoder.fit_transform(full_df['Burnout_Risk'])  # Convert 'Low', 'Medium', 'High' into numbers

# Define features (X) and target (y)
# X = full_df.drop(columns=['Employee_ID', 'Burnout_Risk'])  # Features
selected_features = [
    "Gender", "Age", "Department", "Experience_Level",
    "Tasks_Assigned", "Task_Difficulty", "Hours_Worked",
    "Deadline_Days", "Stress_Level", "Sick_Leaves",
    "Self_Reported_Fatigue", "Productivity_Score"
]
X = full_df[selected_features]  # Use only selected 12 features

y = full_df['Burnout_Risk']  # Target variable

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# Initialize model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("✅ Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy:.2f}")

# Detailed classification report
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Get feature importance
feature_importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10,5))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette='viridis')
plt.title('🔍 Feature Importance in Burnout Prediction')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Define hyperparameters to test
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("✅ Best Parameters:", grid_search.best_params_)

# Train best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"🚀 Optimized Model Accuracy: {accuracy_best:.2f}")

plt.figure(figsize=(12,8))
plot_tree(best_model, feature_names=X.columns, class_names=['Low', 'Medium', 'High'], filled=True, fontsize=7)
plt.title('🌳 Decision Tree Visualization')
plt.show()

# Generate predictions
y_pred = model.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions & Accuracy
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"🚀 Random Forest Model Accuracy: {rf_accuracy:.2f}")

# Save the test dataset
test_data = X_test.copy()
test_data['Burnout_Risk'] = y_test
test_data.to_csv("test_data.csv", index=False)
print("✅ Test data saved as 'test_data.csv'!")

# Save feature encoder
joblib.dump(encoder, "encoder.pkl")
print("✅ Encoder saved successfully!")

# Save the model
joblib.dump(rf_model, "fairwork_ai_model.pkl")
print("✅ Model saved successfully!")

