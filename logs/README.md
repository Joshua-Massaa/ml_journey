# ml_journey
My ml progress. 

This repository documents my machine learning journey from the ground up. Below is a recap of what’s been covered so far:

---

## **Lesson 1 – Kickoff & Setup**

### What I Learned
- What ML is (supervised vs unsupervised)
- The concept of training vs testing
- Decided to work locally with pip + VS Code

### Reflections
Feeling excited. Noticed how many things in ML are just pattern + data. Plan to focus on fundamentals.

---

## **Lesson 2 – First Classifier – Iris Dataset**

### What I Learned
- Built my first classifier to predict iris flower species based on petal length.
- Practiced splitting data, training a model, and evaluating with accuracy.
  
###  Files
- ** File**: `iris_classifier.py`

###  Reflections
The simplicity of classification was refreshing. It was a good start to get familiar with model training and testing.

---

## *Lesson 3 – First Regressor – House Prices**

###  What I Learned
- Built a regression model to predict house prices using the number of rooms as the only feature.
- Learned about **linear regression**, model coefficients, and **Mean Squared Error (MSE)**.

###  Files
- ** File**: `house_price_predictor.py`

### Reflections
The concept of regression makes a lot of sense, and understanding the model coefficients gave me deeper insight into how predictions are made.

---

## **Lesson 4 – Multi-Feature Regression & Overfitting**

###  What I Learned
- Built a regression model using all features in the Boston housing dataset.
- Compared train and test MSE to detect **overfitting**.
- Learned that a good model should generalize well, not just memorize.

### Files
- **File**: `house_price_multifeature.py`

### Reflections
Overfitting is a key challenge to watch for. I’m excited to dive into techniques to address it, like regularization.

---

---

## **Lesson 5 – Ridge Regression & Regularization**

### What I Learned
- Explored **Ridge Regression**, a regularized version of linear regression that addresses overfitting.
- Learned how **alpha** is a hyperparameter that controls the strength of regularization.
- Understood that as alpha increases, the model penalizes large coefficients more heavily, leading to **coefficient shrinkage**.
- Observed the **bias-variance trade-off**: small alpha values allow for flexible models, while large alpha values can oversimplify.

### Visual Analysis
- **Left Plot (MSE vs Alpha)**:  
  MSE remained nearly constant for alpha values from 0.01 to 10, suggesting the model's performance is stable in this range.  
  However, at **alpha = 1000**, MSE increased noticeably, showing the model had become too constrained—this is **underfitting** in action.

- **Right Plot (Coefficient Shrinkage)**:  
  As alpha increases, the magnitude of the coefficients decreases. Some features (like `Latitude`, `Longitude`, and `AveRooms`) shrink more than others, showing how Ridge selectively dampens coefficients based on their influence and correlation.

### Files
- **File**: `ridge_vs_linear_regression.py` `ridge_visual_test.py`_

### Reflections
This lesson helped cement my understanding of **regularization**. The fact that MSE didn’t change much for smaller alphas but jumped at alpha=1000 clearly showed the cost of over-penalization. Visualizing the shrinkage of weights made it easier to grasp why Ridge is useful—it’s not just about performance but also model simplicity and robustness.

---

# Lesson 6 – Polynomial Regression

### What I Learned
- How to create **polynomial features** (like \(x^2\)) to capture curves in data.
- Built both a **Linear Regression** and a **Polynomial Regression** model.
- Saw how simple linear models struggle with non-linear data.
- Learned to use `PolynomialFeatures` from scikit-learn to **expand** inputs smartly.

### Tools Used
- Scikit-learn (`LinearRegression`, `PolynomialFeatures`, `mean_squared_error`)

### Experiments
- Generated random data with a **quadratic** relationship.
- Trained a **Linear Regression** model — it **underfit** (couldn't capture the curve).
- Trained a **Polynomial Regression** model — it **fit the data much better**.
- Compared **MSE** between the two models to see the performance difference.

### Results
- **Polynomial Regression** had a **much lower MSE** than basic Linear Regression.
- Visualizations made it very clear how feature expansion can capture complexity.

### Reflections
- Sometimes "simple" linear models aren't enough.
- Carefully adding new features (like \(x^2\), \(x^3\)) can unlock better predictions.
- I’m starting to see **how model complexity and data shape connect**.

---

# Files Added
- polynomial_regression.py

