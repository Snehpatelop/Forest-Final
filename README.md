# Stock Market Prediction Using XGBoost

This repository demonstrates a machine learning-based approach to predict stock market closing prices using the XGBoost regression algorithm. By analyzing historical stock market data, the model forecasts the next day's closing price, helping users understand potential market trends.

## Features

1. **Data Preprocessing**:
   - Handles missing values and formats the dataset for machine learning.
   - Adds engineered features like moving averages, high-low differences, and open-close differences.

2. **Feature Scaling**:
   - Scales features using `StandardScaler` to enhance model accuracy and convergence.

3. **Model Training**:
   - Utilizes XGBoost for regression, offering fast and accurate predictions.
   - Supports hyperparameter tuning for performance optimization.

4. **Performance Evaluation**:
   - Assesses model accuracy using:
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - R² Score
     - Mean Absolute Percentage Error (MAPE)

5. **Visualizations**:
   - Generates plots comparing actual vs predicted prices.
   - Displays feature importance to highlight key contributors to predictions.

6. **Interactive Prediction**:
   - Allows users to input a specific date and predicts the next day's closing price for that date.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Snehpatelop/STOCK-MARKET-PREDICTION-USING-xgboost.git
   cd STOCK-MARKET-PREDICTION-USING-xgboost
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a dataset in CSV format with the following columns:
   - `Date`: The date in `YYYY-MM-DD` format.
   - `Open`: The stock's opening price.
   - `High`: The highest price of the day.
   - `Low`: The lowest price of the day.
   - `Close`: The stock's closing price.

## Usage

1. Place your dataset in the project directory.
2. Update the file path in the script to point to your dataset.
3. Run the script:
   ```bash
   python stock_price_prediction.py
   ```
4. Follow the prompts to input a date for prediction.

## Example Output

### Metrics
```
Root Mean Squared Error: 50.23
Mean Absolute Error: 42.15
R² Score: 0.95
MAPE: 0.89%
```

### Predictions
Input a specific date (e.g., `12-12-2024`), and the script will display:
```
Predicted Next Day Close Price: 12345.67
Actual Next Day Close Price: 12300.00
```

### Visualizations
- **Actual vs Predicted Prices**: A line chart comparing predictions with actual values.
- **Feature Importance**: A bar chart showing the importance of each feature used in the model.

- ![image](https://github.com/user-attachments/assets/17cfe01b-fbb9-4a31-b030-32973eab3fb3)
![image](https://github.com/user-attachments/assets/ce682f17-3cac-4720-84f5-46a85773859a)


## Dependencies

The required libraries for this project are listed in the `requirements.txt` file. Install them using:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Potential Improvements

- Incorporate additional features like RSI, MACD, or Bollinger Bands.
- Add trading volume as a predictive feature.
- Deploy the model as a web application using Flask or Streamlit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

