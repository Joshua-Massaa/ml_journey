import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Explore the data
print("First 5 rows of the dataset:")
print(df.head())
print("\nSummary stats:")
print(df.describe())

# Visualize pair relationships
sns.pairplot(df, hue='species')
plt.suptitle("Iris Data Pair Plot", y=1.02)
plt.tight_layout()
plt.show()

# Prepare features and labels
X = df.drop('species', axis=1)
y = df['species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy * 100:.2f}%")
