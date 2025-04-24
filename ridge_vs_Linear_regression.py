from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
linear = LinearRegression()
ridge = Ridge(alpha=100) #tested alpha = 1.0, 10.0, 0.1, 0.01

linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)

# Evaluate
linear_mse = mean_squared_error(y_test, linear.predict(X_test))
ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))

print(f"Linear Regression MSE: {linear_mse:.2f}")
print(f"Ridge Regression MSE: {ridge_mse:.2f}")
