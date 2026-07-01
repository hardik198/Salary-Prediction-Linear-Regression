# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Program Started")

# Load Dataset
df = pd.read_csv("Salary_dataset.csv")

# Display Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df[["YearsExperience"]]
y = df["Salary"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Model Parameters
print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict Test Data
y_pred = model.predict(X_test)

# Predict New Salary
experience = [[6.5]]
predicted_salary = model.predict(experience)

print(f"\nPredicted Salary for 6.5 Years Experience: ₹{predicted_salary[0]:,.2f}")

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_train, model.predict(X_train), label="Regression Line")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Compare Actual vs Predicted
result = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nActual vs Predicted")
print(result)

# Save Model
joblib.dump(model, "salary_prediction_model.pkl")

# Load Model
loaded_model = joblib.load("salary_prediction_model.pkl")

salary = loaded_model.predict([[10]])

print(f"\nPredicted Salary for 10 Years Experience: ₹{salary[0]:,.2f}")