import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/snehj/OneDrive/Desktop/Forest final/bank-nifty-1h-data (1).csv")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
print("Dataset Head:")
print(data.head())

# Feature engineering
data['High-Low'] = data['High'] - data['Low']
data['Open-Close'] = data['Open'] - data['Close']
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# Target variable: Shift to predict the next day's close
data['Next_Day_Close'] = data['Close'].shift(-1)

# Drop rows with NaN values
data = data.dropna()

# Define features and target
features = ['Open', 'High', 'Low', 'High-Low', 'Open-Close', 'MA_10', 'MA_50']
X = data[features]
y = data['Next_Day_Close']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Visualization for model performance
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label="Actual Next Day Price", color="blue")
plt.plot(y_pred[:50], label="Predicted Next Day Price", color="red")
plt.title("Next Day Stock Price Prediction")
plt.xlabel("Data Points")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

importances = model.feature_importances_
features_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(features_importance_df['Feature'], features_importance_df['Importance'])
plt.xticks(rotation=45)
plt.title('Feature Importances')
plt.show()
# Visualization to check for overfitting
plt.figure(figsize=(10, 5))
plt.plot(model.predict(X_train)[:50], label="Training Predictions", color="green")
plt.plot(y_train.values[:50], label="Training Actuals", color="orange")
plt.plot(y_test.values[:50], label="Testing Actuals", color="blue")
plt.plot(y_pred[:50], label="Testing Predictions", color="red")
plt.title("Overfitting Check: Training vs Testing Predictions")
plt.xlabel("Data Points")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


# Prediction for user-specified date
# Prediction for user-specified date
# Prediction for user-specified date
# Prediction for user-specified date
# Prediction for user-specified date
# Prediction for user-specified date
# Prediction for user-specified date
# Convert the input date to datetime
# Prompt the user for input date
# Prompt the user for input date
input_date = input("Enter the date (DD-MM-YYYY) to predict the next day's stock price: ")

try:
    # Convert the input date to datetime
    input_date_dt = pd.to_datetime(input_date, dayfirst=True)

    # Check if the date exists in the dataset
    if input_date_dt not in data.index:
        print(f"Date {input_date} is not in the dataset.")
    else:
        # Extract the features for the input date as a Series
        features_for_date = data.loc[input_date_dt, features]
        
        # If there are multiple rows for the date, select the first one explicitly
        if isinstance(features_for_date, pd.DataFrame):
            features_for_date = features_for_date.iloc[0]
        
        # Reshape and scale the extracted features
        features_for_date = features_for_date.values.reshape(1, -1)  # Reshape to (1, 7)
        scaled_features_for_date = scaler.transform(features_for_date)
        
        # Predict the next day's stock price
        predicted_price = model.predict(scaled_features_for_date)[0]
        
        # Use shift to get the actual price for the next day (if it exists)
        next_day_data = data.shift(-1)
        next_day_actual_price = next_day_data.loc[input_date_dt, 'Close'] if input_date_dt in next_day_data.index else None
        
        if next_day_actual_price is not None:
            # Access the scalar value from the Series before formatting
            next_day_actual_price = next_day_actual_price.iloc[0] if isinstance(next_day_actual_price, pd.Series) else next_day_actual_price
            print(f"Predicted Next Day Close Price: {predicted_price:.2f}")
            print(f"Actual Next Day Close Price: {next_day_actual_price:.2f}")
        else:
            print("No data available for the next day to compare.")

except Exception as e:
    print(f"An error occurred: {e}")
