# Build a multivariate linear regression model that can predict the product sales based on the advertising budget allocated to different channels. 
# The features are TV Budget ($),Radio Budget ($),Newspaper Budget ($) and the output is Sales (units)

import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([
    [230.1, 37.8, 69.2],
    [44.5, 39.3, 45.1],
    [17.2, 45.9, 69.3],
    [151.5, 41.3, 58.5],
    [180.8, 10.8, 58.4],
    [8.7, 48.9, 75.0],
    [57.5, 32.8, 23.5],
    [120.2, 19.6, 11.6],
    [144.1, 16.0, 40.3],
    [111.6, 12.6, 37.9]
])

y = np.array([22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 15.6, 12.2])

# Train model
model = LinearRegression()
model.fit(X, y)

print("Multivariate Linear Regression Model is Ready!")

# Take user input
tv = float(input("Enter TV Budget ($): "))
radio = float(input("Enter Radio Budget ($): "))
newspaper = float(input("Enter Newspaper Budget ($): "))

# Predict sales
predicted_sales = model.predict([[tv, radio, newspaper]])[0]

print(f"\nPredicted Sales (units): {predicted_sales:.2f}")