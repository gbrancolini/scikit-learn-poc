import numpy as np
from sklearn.linear_model import Ridge

# Sample import features (house areas and number of bedrooms)
X = np.array([[100,2], [150, 3], [200, 3], [250, 4], [300, 4]])

# Correspoding target variable (house prices)
y = np.array([250000, 350000, 450000, 550000, 650000])

model = Ridge(alpha =0.5) #alpha is the regularization strength

model.fit(X, y)

new_house_features = np.array([[175, 2], [225, 3]])
predicted_prices = model.predict(new_house_features)

for features, price in zip(new_house_features, predicted_prices):
    print(f"House Area, {features[0]} sqft, Badrooms: {features[1]}, Predicted Price, ${price: .2f}")