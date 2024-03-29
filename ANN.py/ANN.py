import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
dry_bean_dataset = fetch_ucirepo(id=602)
X = dry_bean_dataset.data.features
y = dry_bean_dataset.data.targets
y = y.values.ravel()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                      solver='adam', verbose=10, random_state=42, tol=0.0001)

model.fit(X_train_scaled, y_train)

# Predictions
predictions = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f'Test Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
