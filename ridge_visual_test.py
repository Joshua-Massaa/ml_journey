import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load and scale data
housing = fetch_california_housing()
X, y = housing.data, housing.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Test multiple alpha values
alphas = [0.01, 0.1, 1, 10, 100, 1000]
mses = []
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    mses.append(mean_squared_error(y_test, preds))
    coefs.append(ridge.coef_)

# Plot MSE vs Alpha
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(alphas, mses, marker='o')
plt.xscale('log')
plt.xlabel("Alpha (log scale)")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Alpha")

# Plot Coefficient shrinkage
plt.subplot(1, 2, 2)
coefs = np.array(coefs)
for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], marker='o', label=housing.feature_names[i])

plt.xscale('log')
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient Value")
plt.title("Coefficient Shrinkage with Ridge")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()
