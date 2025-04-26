
#impoorting libraries 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#generating fake data 

np.random.seed(42)

# Features (input)
X = np.random.rand(100, 1) * 6 - 3  # Random numbers between -3 and 3

# Targets (output)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)  # Quadratic relationship + some noise

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predict
y_pred_linear = lin_reg.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred_linear, color='red', label='Linear Prediction')
plt.legend()
plt.title("Linear Regression on Curvy Data")
plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Predict
y_pred_poly = poly_reg.predict(X_poly)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, y_pred_poly, color='green', label='Polynomial Prediction', s=10)
plt.legend()
plt.title("Polynomial Regression (Degree 2)")
plt.show()

#Evaluate
mse_linear = mean_squared_error(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"MSE Linear Regression: {mse_linear:.4f}")
print(f"MSE Polynomial Regression: {mse_poly:.4f}")

# Degree 5 Polynomial
poly_features_5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly_5 = poly_features_5.fit_transform(X)

poly_reg_5 = LinearRegression()
poly_reg_5.fit(X_poly_5, y)

y_pred_poly_5 = poly_reg_5.predict(X_poly_5)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, y_pred_poly_5, color='purple', label='Degree 5 Prediction', s=10)
plt.legend()
plt.title("Polynomial Regression (Degree 5)")
plt.show()

# Evaluate
mse_poly_5 = mean_squared_error(y, y_pred_poly_5)
print(f"MSE Polynomial Regression (Degree 5): {mse_poly_5:.4f}")

