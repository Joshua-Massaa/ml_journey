import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset from a CSV URL (Boston housing dataset alternative)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Print the first few rows and basic stats
print("Dataset preview:")
print(df.head())
print("\nSummary stats:")
print(df.describe())

# Plot relationship between RM (average rooms) and MEDV (house price)
sns.scatterplot(data=df, x='rm', y='medv')
plt.title("Average Rooms vs. Median House Price")
plt.show()

# Features and label
X = df[['rm']]  # just one feature: average number of rooms
y = df['medv']  # target: median value of owner-occupied homes

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"Model Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Model Intercept (bias): {model.intercept_:.2f}")
