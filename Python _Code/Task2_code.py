import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data (replace with your actual paths)
train_data = pd.read_csv("DATASETS/train - train.csv")
test_data = pd.read_csv("DATASETS/test - test.csv")  # Load test data separately

# Separate features and target
X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

# Define multiple models
models = [
    ("Logistic Regression", LogisticRegression()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
]

# Store predictions and accuracies for each model
all_predictions = {}
train_accuracies = []

for name, model in models:
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    train_accuracies.append(train_accuracy)
    print(f"{name} Training Accuracy: {train_accuracy:.4f}")  # Print training accuracy

    predictions = model.predict(test_data)  # Use test_data for predictions
    all_predictions[name] = predictions

# Create a DataFrame with predictions
all_results = pd.DataFrame(all_predictions)

# Save the DataFrame to a CSV file
all_results.to_csv("test_predictions1.csv", index=False)  # Adjust filename as needed
