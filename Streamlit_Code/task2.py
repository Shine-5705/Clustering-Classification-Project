import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load default training data (replace with your actual path if needed)

st.header("Exploring Classification Models & Predicting target values")
train_data_path = "/home/shinegupta/Documents/deployement/train - train.csv"
test_data_path = st.text_input("Enter path for test data:", "/home/shinegupta/Documents/deployement/test - test.csv")

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)  # Load test data separately

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

st.markdown(f"<p style='font-size:20px; font-weight:bold;'>Accuracies of Different models :</p>", unsafe_allow_html=True)


for name, model in models:
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    train_accuracies.append(train_accuracy)
    st.write(f"{name} Training Accuracy: {train_accuracy:.4f}")  # Print training accuracy

    predictions = model.predict(test_data)  # Use test_data for predictions
    all_predictions[name] = predictions

st.subheader("Explanation of Algorithm Choices")
st.markdown("""
- **Logistic Regression**:
  - It's a fundamental algorithm for binary classification tasks.
  - Simple and interpretable, it's suitable for linear relationships between features and target.

- **Decision Tree**:
  - Non-parametric and able to capture non-linear relationships.
  - Provides insights into feature importance and decision-making processes.

- **Random Forest**:
  - An ensemble learning method based on Decision Trees.
  - Handles non-linear relationships, high-dimensional data, and noisy data effectively.
  - Provides estimates of feature importance for better understanding of the data.
""")

# Create a DataFrame with predictions
all_results = pd.DataFrame(all_predictions)

# Save the DataFrame to a CSV file
filename = st.text_input("Enter filename for saving predictions:", "test_predictions.csv")
all_results.to_csv(filename, index=False)  # Adjust filename as needed
