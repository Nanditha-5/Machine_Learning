# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load & Preprocess Dataset
# -----------------------------
column_names = [
    'Gender', 'Age', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 
    'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Learner'
]

df = pd.read_csv("learning-style-dataset.csv", names=column_names)

# Encode features
feature_columns = column_names[:-1]
label_encoders = {}
for col in feature_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
df["Learner"] = target_le.fit_transform(df["Learner"])

# Prepare training data
X = df[feature_columns]
y = df["Learner"]

# -----------------------------
# Compare Models (Cross-Validation)
# -----------------------------
print("\n--- Cross-Validation Accuracy Comparison ---")
dt_model = DecisionTreeClassifier()
dt_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
print(f"Decision Tree CV Accuracy: {dt_scores.mean():.4f} (+/- {dt_scores.std():.4f})")

gb_model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
gb_scores = cross_val_score(gb_model, X, y, cv=5, scoring='accuracy')
print(f"Gradient Boosting CV Accuracy: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")

# Optional Visualization
models = ['Decision Tree', 'Gradient Boosting']
means = [dt_scores.mean(), gb_scores.mean()]
stds = [dt_scores.std(), gb_scores.std()]

plt.bar(models, means, yerr=stds, capsize=10, color=['orange', 'green'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison (5-fold CV)')
plt.savefig("model_accuracy_comparison.png")  # Save for report/presentation

# -----------------------------
# Train Final Model
# -----------------------------
model = gb_model  # Using the better-performing ensemble model
model.fit(X, y)

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        # Prepare input data
        input_data = []
        input_data.append(data['gender'])
        input_data.append(str(data['age']))

        answers = data['answers']
        if len(answers) < 15:
            answers = answers + ['1'] * (15 - len(answers))
        elif len(answers) > 15:
            answers = answers[:15]
        input_data.extend(answers)

        print("Processed input data:", input_data)

        # Encode input
        encoded_input = []
        for col, val in zip(feature_columns, input_data):
            val_str = str(val)
            if val_str in label_encoders[col].classes_:
                encoded_val = label_encoders[col].transform([val_str])[0]
            else:
                most_common = df[col].mode()[0]
                encoded_val = most_common
                print(f"Warning: Value '{val_str}' not in encoder for '{col}'. Using most common.")
            encoded_input.append(encoded_val)

        print("Encoded input:", encoded_input)

        # Predict
        prediction = model.predict([encoded_input])
        predicted_label = target_le.inverse_transform(prediction)

        print("Prediction:", predicted_label[0])
        return jsonify({"learning_style": predicted_label[0]})
    
    except Exception as e:
        print("Error in prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
