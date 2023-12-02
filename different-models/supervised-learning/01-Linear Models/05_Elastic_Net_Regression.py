import numpy as np
from sklearn.linear_model import ElasticNet

# Sample input features (house areas and number of bedrooms)
X = np.array([[100, 2], [150, 3], [200, 3], [250, 4], [300, 4]])

# Corresponding target variable (house prices)
y = np.array([250000, 350000, 450000, 550000, 650000])

# Create an Elastic Net regression model
model = ElasticNet(alpha=0.1, l1_ratio=0.6)  # Alpha and l1_ratio are regularization hyperparameters

# Train the model
model.fit(X, y)

# Make predictions for new data
new_house_features = np.array([[175, 2], [225, 3]])
predicted_prices = model.predict(new_house_features)

# Print the predicted prices
for features, price in zip(new_house_features, predicted_prices):
    print(f"House Area: {features[0]} sqft, Bedrooms: {features[1]}, Predicted Price: ${price:.2f}")
