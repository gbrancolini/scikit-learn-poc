import numpy as np
from sklearn.linear_model import MultiTaskLasso

# Sample input features (house areas and number of bedrooms)
X = np.array([[100, 2], [150, 3], [200, 3], [250, 4], [300, 4]])

# Corresponding target variables for two tasks (house prices and rents)
y = np.array([[200000, 2000], [350000, 2500], [450000, 3000], [550000, 3500], [650000, 4000]])

# Create a Multi-task Lasso regression model
model = MultiTaskLasso(alpha=0.1) # Alpha is the regularization strengh

# Train the model
model.fit(X, y)

# Make prediction for new data
new_houses_features = np.array([[175, 2], [225, 3]])
predicted_values = model.predict(new_houses_features)

for features, values in zip(new_houses_features, predicted_values):
    price, rent = values
    print(f"House Area: {features[0]} sqft, Bedrooms: {features[1]}")
    print(f"Predicted Price: ${price:.2f}, Predicted Rent: ${rent: .2f}")
    print()