import numpy as np
from sklearn.linear_model import Lars

# Sample input features (house areas and number of bedrooms)
X = np.array([[100, 2], [150, 3], [200, 3], [250, 4], [300, 4]])

# Corresponding target variable (house prices)
y = np.array([250000, 350000, 450000, 550000, 650000])

# Create a LARS regression model
model = Lars(n_nonzero_coefs=2)  # Limit the model to use at most 2 features

# Train the model
model.fit(X, y)

# Make predictions for new data
new_house_features = np.array([[175, 2], [225, 3]])
predicted_prices = model.predict(new_house_features)

# Print the predicted prices
for features, price in zip(new_house_features, predicted_prices):
    print(f"House Area: {features[0]} sqft, Bedrooms: {features[1]}, Predicted Price: ${price:.2f}")
