import numpy as np
from sklearn.linear_model import LinearRegression

# Sample input features (house areas)
X = np.array([[100], [150], [200], [250], [300]])

# Corresponding target variable (house prices)
Y = np.array([250000, 350000, 450000, 550000, 650000])

# Create a linear regreesion model
model = LinearRegression()

# Train the model
model.fit(X,Y)

# Make predictions for new data
new_house_areas = np.array([[175], [225]])
predicted_prices = model.predict(new_house_areas)

# Print the predicted prices
for area, price in zip(new_house_areas, predicted_prices):
    print(f"House Area: {area[0]} sqft, Predicted Price: ${price:.2f}")