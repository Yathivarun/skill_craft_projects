import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:/Users/pdx/OneDrive/Desktop/mini_project/z_internship/datasets/kc_house_data.csv")

# Use relevant features
features = df[['sqft_living', 'bedrooms', 'bathrooms']]
target = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
print("Training R²:", r2_score(y_train, y_train_pred))
print("Testing R²:", r2_score(y_test, y_test_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# ---------- GRAPHS ---------- #

plt.figure(figsize=(18, 12))

# 1. Train Actual vs Predicted
plt.subplot(2, 2, 1)
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Train Data: Actual vs Predicted")

# 2. Test Actual vs Predicted (color-coded correct/wrong)
plt.subplot(2, 2, 2)
threshold = 100000  # define how far off is considered "wrong"
diff = np.abs(y_test - y_test_pred)
correct = diff <= threshold

sns.scatterplot(x=y_test, y=y_test_pred, hue=correct.map({True: 'Correct', False: 'Wrong'}), alpha=0.6, palette={'Correct': 'green', 'Wrong': 'red'})
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Test Data: Correct vs Wrong Predictions")
plt.legend(title="Prediction Status")

# 3. Residual Histogram
plt.subplot(2, 2, 3)
residuals = y_test - y_test_pred
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals (Test Data)")
plt.xlabel("Error")
plt.ylabel("Frequency")

# 4. Correlation Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(df[['price', 'sqft_living', 'bedrooms', 'bathrooms']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with Price")

plt.tight_layout()
plt.show()

# ---------- USER PREDICTION ---------- #

print("\n--- Predict Your Own House Price ---")
try:
    sqft = float(input("Enter square footage (e.g. 500 - 10000): "))
    sqft = max(300, sqft)  # clamp to minimum realistic range

    bed = int(input("Enter number of bedrooms (e.g. 1 - 10): "))
    bath = float(input("Enter number of bathrooms (e.g. 1 - 8): "))

    custom_input = pd.DataFrame([[sqft, bed, bath]], columns=['sqft_living', 'bedrooms', 'bathrooms'])
    predicted_price = model.predict(custom_input)[0]
    print(f"Estimated House Price: ${predicted_price:,.2f}")

except ValueError:
    print("Invalid input! Please enter numerical values.")
