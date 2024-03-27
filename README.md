# AI-powered-Traffic-Optimization
利用机器学习和边缘计算优化城市交通，减少拥堵并提高道路安全。
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Simulated data generation
# Features: hour of day (0-23), day of week (0-6), weather (0: clear, 1: rain, 2: snow), is_holiday (0: no, 1: yes)
# Target: congestion level (0-100)
np.random.seed(42)
X = np.random.randint(0, 4, (1000, 4))
y = np.random.randint(0, 101, (1000,))

# Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and model evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Visualization of actual vs. predicted congestion levels
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel('Actual congestion level')
plt.ylabel('Predicted congestion level')
plt.title('Actual vs. Predicted Congestion Levels')
plt.plot([0, 100], [0, 100], 'k--') # Reference line
plt.show()

# Example of making a prediction
# Hour: 8 AM, Monday, Clear Weather, Not a Holiday
example_feature = np.array([[8, 1, 0, 0]])
predicted_congestion = model.predict(example_feature)
print(f"Predicted congestion level for the given conditions: {predicted_congestion[0]}")

# Note: This is a simplified demo. Real-world applications would require more complex models, larger datasets,
# and integration with edge computing devices for real-time data processing and prediction.
