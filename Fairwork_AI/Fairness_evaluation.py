import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the trained model
rf_model = joblib.load("fairwork_ai_model.pkl")
encoder = joblib.load("encoder.pkl")

# Load and merge datasets
full_df = pd.read_csv('employee_demographics.csv').merge(
    pd.read_csv('work_data.csv'), on='Employee_ID').merge(
    pd.read_csv('stress_burnout.csv'), on='Employee_ID')


# Ensure 'Burnout_Risk' is encoded
full_df['Burnout_Risk'] = encoder.fit_transform(full_df['Burnout_Risk'])

# Split dataset
X = full_df.drop(columns=['Employee_ID', 'Burnout_Risk'])
y = full_df['Burnout_Risk']

# Define categorical and numerical features
categorical_features = ['Gender', 'Department', 'Experience_Level']
numerical_features = X.columns.difference(categorical_features)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Save the trained model
joblib.dump(rf_model, "fairwork_ai_model.pkl")

# Make predictions
y_pred = rf_model.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy:.2f}")

# Classification report
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Function to evaluate fairness and detect bias
def evaluate_fairness_by_group(df, group_column):
    group_accuracies = {}

    for group in df[group_column].unique():
        group_data = df[df[group_column] == group]
        X_group = group_data.drop(columns=['Employee_ID', 'Burnout_Risk'])
        y_group = group_data['Burnout_Risk']

        # Apply the same preprocessing
        X_group_processed = preprocessor.transform(X_group)

        y_group_pred = rf_model.predict(X_group_processed)
        group_accuracy = accuracy_score(y_group, y_group_pred)
        group_accuracies[group] = group_accuracy

    return group_accuracies

# Evaluate fairness for Gender, Department, and Experience Level
gender_fairness = evaluate_fairness_by_group(full_df, 'Gender')
department_fairness = evaluate_fairness_by_group(full_df, 'Department')
experience_fairness = evaluate_fairness_by_group(full_df, 'Experience_Level')

# Print fairness results
print(f"📊 Gender Fairness (Accuracy by Group): {gender_fairness}")
print(f"📊 Department Fairness (Accuracy by Group): {department_fairness}")
print(f"📊 Experience Level Fairness (Accuracy by Group): {experience_fairness}")

# Function to check if bias exists
def detect_bias(fairness_dict, category_name):
    min_acc = min(fairness_dict.values())
    max_acc = max(fairness_dict.values())
    
    if (max_acc - min_acc) > 0.10:  # If accuracy difference is greater than 10%
        print(f"⚠️ **Bias Detected in {category_name}**: Difference = {max_acc - min_acc:.2f}")
        print(f"🔎 **Biased Towards**: {max(fairness_dict, key=fairness_dict.get)}")
        print(f"❌ **Biased Against**: {min(fairness_dict, key=fairness_dict.get)}\n")
        return True
    else:
        print(f"✅ **No significant bias detected in {category_name}**.\n")
        return False

# Check for bias
is_gender_biased = detect_bias(gender_fairness, "Gender")
is_department_biased = detect_bias(department_fairness, "Department")
is_experience_biased = detect_bias(experience_fairness, "Experience Level")

# Final conclusion
if is_gender_biased or is_department_biased or is_experience_biased:
    print("🚨 **Fairness Alert!** Your model is biased in one or more categories.")
else:
    print("🎉 **Your model is fair across all groups!**")

# Save results
fairness_results = {
    'Gender': gender_fairness,
    'Department': department_fairness,
    'Experience_Level': experience_fairness
}

pd.DataFrame(fairness_results).to_csv("fairness_evaluation_results.csv", index=False)
print("✅ Fairness results saved to 'fairness_evaluation_results.csv'")
