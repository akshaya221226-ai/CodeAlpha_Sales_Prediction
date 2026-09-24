# ================================================================
# CODEALPHA TASK 4 - SALES PREDICTION USING PYTHON
# ================================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================================================
# 2. PROJECT TITLE
# ================================================================

print("=" * 70)
print("              SALES PREDICTION USING PYTHON")
print("=" * 70)


# ================================================================
# 3. LOAD DATASET
# ================================================================

file_path = r"C:\Users\Hp\Downloads\archive (7)\Advertising.csv"
df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")


# ================================================================
# 4. DISPLAY DATASET INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ================================================================
# 5. DATA CLEANING
# ================================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Remove unnecessary column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove duplicate rows
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

print("\nData cleaning completed successfully!")

print("\nDataset after cleaning:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)


# ================================================================
# 6. STATISTICAL ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

print(df.describe())


# ================================================================
# 7. FEATURE SELECTION
# ================================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

# Input features
X = df[["TV", "Radio", "Newspaper"]]

# Target
y = df["Sales"]

print("\nInput Features:")
print("TV")
print("Radio")
print("Newspaper")

print("\nTarget Variable:")
print("Sales")


# ================================================================
# 8. TRAIN TEST SPLIT
# ================================================================

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ================================================================
# 9. CREATE AND TRAIN MODEL
# ================================================================

print("\n" + "=" * 70)
print("TRAINING LINEAR REGRESSION MODEL")
print("=" * 70)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ================================================================
# 10. MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(X_test)

print("\nSales predictions generated successfully!")


# ================================================================
# 11. MODEL PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"R2 Accuracy: {r2 * 100:.2f}%")


# ================================================================
# 12. ADVERTISING IMPACT
# ================================================================

print("\n" + "=" * 70)
print("ADVERTISING IMPACT ON SALES")
print("=" * 70)

print(f"\nTV coefficient: {model.coef_[0]:.4f}")
print(f"Radio coefficient: {model.coef_[1]:.4f}")
print(f"Newspaper coefficient: {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# ================================================================
# 13. SAVE PREDICTIONS
# ================================================================

prediction_results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

prediction_results.to_csv(
    "sales_predictions.csv",
    index=False
)

print("\nPrediction file created:")
print("sales_predictions.csv")


# ================================================================
# 14. CREATE REPORT
# ================================================================

with open("sales_prediction_report.txt", "w") as report:

    report.write("CODEALPHA - SALES PREDICTION USING PYTHON\n")
    report.write("=" * 60 + "\n\n")

    report.write("Dataset Information\n")
    report.write("-" * 60 + "\n")
    report.write(f"Total samples: {len(df)}\n")
    report.write("Features: TV, Radio, Newspaper\n")
    report.write("Target: Sales\n\n")

    report.write("Machine Learning Model\n")
    report.write("-" * 60 + "\n")
    report.write("Linear Regression\n\n")

    report.write("Model Performance\n")
    report.write("-" * 60 + "\n")
    report.write(f"MAE: {mae:.4f}\n")
    report.write(f"MSE: {mse:.4f}\n")
    report.write(f"RMSE: {rmse:.4f}\n")
    report.write(f"R2 Score: {r2:.4f}\n")
    report.write(f"R2 Accuracy: {r2 * 100:.2f}%\n\n")

    report.write("Advertising Coefficients\n")
    report.write("-" * 60 + "\n")
    report.write(f"TV: {model.coef_[0]:.4f}\n")
    report.write(f"Radio: {model.coef_[1]:.4f}\n")
    report.write(f"Newspaper: {model.coef_[2]:.4f}\n")
    report.write(f"Intercept: {model.intercept_:.4f}\n")

print("Report created:")
print("sales_prediction_report.txt")


# ================================================================
# 15. GRAPH 1 - SALES DISTRIBUTION
# ================================================================

plt.figure(figsize=(8, 5))

plt.hist(df["Sales"], bins=15, edgecolor="black")

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("01_sales_distribution.png", dpi=300)
plt.close()


# ================================================================
# 16. GRAPH 2 - TV VS SALES
# ================================================================

plt.figure(figsize=(8, 5))

plt.scatter(df["TV"], df["Sales"])

plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("02_tv_vs_sales.png", dpi=300)
plt.close()


# ================================================================
# 17. GRAPH 3 - RADIO VS SALES
# ================================================================

plt.figure(figsize=(8, 5))

plt.scatter(df["Radio"], df["Sales"])

plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("03_radio_vs_sales.png", dpi=300)
plt.close()


# ================================================================
# 18. GRAPH 4 - NEWSPAPER VS SALES
# ================================================================

plt.figure(figsize=(8, 5))

plt.scatter(df["Newspaper"], df["Sales"])

plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("04_newspaper_vs_sales.png", dpi=300)
plt.close()


# ================================================================
# 19. GRAPH 5 - ACTUAL VS PREDICTED
# ================================================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual Sales vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.tight_layout()
plt.savefig("05_actual_vs_predicted.png", dpi=300)
plt.close()


# ================================================================
# 20. GRAPH 6 - ADVERTISING IMPACT
# ================================================================

features = ["TV", "Radio", "Newspaper"]
coefficients = model.coef_

plt.figure(figsize=(8, 5))

plt.bar(features, coefficients)

plt.title("Impact of Advertising Channels on Sales")
plt.xlabel("Advertising Channel")
plt.ylabel("Regression Coefficient")

plt.tight_layout()
plt.savefig("06_advertising_impact.png", dpi=300)
plt.close()


# ================================================================
# 21. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("""
1. 01_sales_distribution.png
2. 02_tv_vs_sales.png
3. 03_radio_vs_sales.png
4. 04_newspaper_vs_sales.png
5. 05_actual_vs_predicted.png
6. 06_advertising_impact.png
7. sales_predictions.csv
8. sales_prediction_report.txt
""")

print("=" * 70)
print("       SALES PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 70)