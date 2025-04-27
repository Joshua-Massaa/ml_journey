import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

#generating fake data
np.random.seed(42)
X = np.linspace(0,5,100).reshape(-1,1)
y = X**2 + np.random.randn(100,1)*1.0  #adding some noise

#creating the models
degree = 5

#Polynomial Regression, No Regularization
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X, y)

#Polynomial Regression, Ridge Regularization
ridge_model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
ridge_model.fit(X, y)

# plot results 
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
plt.scatter(X, y, color='black', label='Noisy Data')

#Predictions 
plt.plot(X_plot, poly_model.predict(X_plot), color='blue', label='Polynomial Regression')
plt.plot(X_plot, ridge_model.predict(X_plot), color='red', label='Ridge Regression')

plt.legend()
plt.title("Normal Polynomial Regression vs Ridge Polynomial Regression")
plt.xlabel("X")
plt.show()


