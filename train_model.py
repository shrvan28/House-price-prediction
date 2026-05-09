import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
# Using relative path assuming this script is in the same directory as the CSV
try:
    df = pd.read_csv("Housing - Housing.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Housing - Housing.csv' not found. Please ensure it's in the same directory.")
    exit()

# Preprocessing (following the notebook's logic)
if 'Address' in df.columns:
    df.drop('Address', axis=1, inplace=True)

# Split features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("Model saved successfully as 'model.pkl'.")

# Save feature names for Streamlit
joblib.dump(list(X.columns), "features.pkl")
print("Feature names saved as 'features.pkl'.")
