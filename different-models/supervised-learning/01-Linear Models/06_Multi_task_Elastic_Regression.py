import numpy as np
from sklearn.linear_model import MultiTaskElasticNet

# Sample input features (house areas and number of bedrooms)
X = np.array([[100, 2], [150, 3], [200, 3], [250, 4], [300, 4]])

# Corresponding target variables for two tasks (house prices and rents)
y = np.array([[250000, 2000], [350000, 2500], [450000, 3000], [550000, 3500], [650000, 4000]])

# Create a Multi-task Elastic Net regression model
model = MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5)  # Alpha and l1_ratio are regularization hyperparameters

# Train the model
model.fit(X, y)

# Make predictions for new data
new_house_features = np.array([[175, 2], [225, 3]])
predicted_values = model.predict(new_house_features)

# Print the predicted prices and rents
for features, values in zip(new_house_features, predicted_values):
    price, rent = values
    print(f"House Area: {features[0]} sqft, Bedrooms: {features[1]}")
    print(f"Predicted Price: ${price:.2f}, Predicted Rent: ${rent:.2f}")
    print()
