# 	1.	Uses sample data (10 houses: sqft, bedrooms, price).
#   2.	Trains a LinearRegression model.
#   3.	Prints the learned equation.
#   4.	Draws a 3D scatter plot of the actual data points.
#   5.	Overlays a regression plane showing predictions.
# Demo: 3D scatter plot + regression plane for house price prediction

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1) Sample Data (synthetic)
# -------------------------------
# Feature 1: Square Feet
sqft = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000])

# Feature 2: Bedrooms
beds = np.array([2, 2, 3, 3, 4, 3, 4, 4, 5, 5])

# Target: Price (in $)
price = np.array([100000, 120000, 150000, 180000, 210000,
                  220000, 250000, 270000, 300000, 330000])

# Combine features into X
X = np.column_stack([sqft, beds])

# -------------------------------
# 2) Train Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X, price)

print("Model Equation:")
print(f"Price = {model.intercept_:.2f} "
      f"+ {model.coef_[0]:.2f} * SqFt "
      f"+ {model.coef_[1]:.2f} * Beds")

# -------------------------------
# 3) Prepare Regression Plane
# -------------------------------
sqft_grid = np.linspace(min(sqft), max(sqft), 30)
beds_grid = np.linspace(min(beds), max(beds), 30)
S, B = np.meshgrid(sqft_grid, beds_grid)

# Predict prices on grid
grid_points = np.column_stack([S.ravel(), B.ravel()])
P = model.predict(grid_points).reshape(S.shape)

# -------------------------------
# 4) Plot
# -------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter actual data
ax.scatter(sqft, beds, price, c="blue", marker="o", s=50, label="Actual Data")

# Regression plane
ax.plot_surface(S, B, P, alpha=0.5, color="orange", label="Prediction Plane")

ax.set_xlabel("Square Feet")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price ($)")
ax.set_title("3D House Price Prediction (Sample Data)")

plt.show()