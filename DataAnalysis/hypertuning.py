import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_vif(data, exclude_columns):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the dataset, excluding specified columns.
    
    :param data: Pandas DataFrame containing the dataset
    :param exclude_columns: List of columns to be excluded from VIF calculation
    :return: DataFrame with features and their respective VIF values
    """
    data_numeric = data.drop(exclude_columns, axis=1)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = data_numeric.columns
    vif_data['VIF'] = [variance_inflation_factor(data_numeric.values, i) for i in range(data_numeric.shape[1])]
    return vif_data

# Load the dataset
data = pd.read_csv('Monthly_Price_Determinant_Data_v4.csv')
logging.info("Dataset loaded successfully.")

# Drop the first row if necessary
data = data.drop(data.index[0])

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')

# Define the target variable
target_variable = 'Iron ore, cfr spot ($/dmtu)'

# Shift the target variable down by one row to align it with the next month's features
data[target_variable] = data[target_variable].shift(-1)

# Drop the last row because it will have NaN after shifting
data = data.dropna()

# Drop the Date column and target variable column before scaling
data_numeric = data.drop(['Date'], axis=1)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Create a DataFrame with the standardized data
data_scaled_df = pd.DataFrame(data_scaled, columns=data_numeric.columns)

# Calculate initial VIF
vif_data_pre = calculate_vif(data_scaled_df, [target_variable])
print('Calculated VIF:')
print(vif_data_pre)

# Remove High VIF features
high_vif_features = ['Trade (% of GDP)', 'Inflation, consumer prices (annual %)',
                     'Output of Electricity Current Period(100 million kwh)',
                     'Total Value of Imports Current Period(1000 US dollars)',
                     'Total Value of Imports and Exports Growth Rate (The same period last year=100)(%)',
                     'Industry (including construction), value added (% of GDP)',
                     'Gross Domestic Product Current Quarter(100 million yuan)']

data_low_vif_df = data_scaled_df.drop(columns=high_vif_features)

# Recalculate VIF
vif_data_post = calculate_vif(data_low_vif_df, [target_variable])
print('Recalculated VIF:')
print(vif_data_post)
print(f'Final Selected Features are: {list(data_low_vif_df.columns)}')

# Filter the original data to be used for training
train_data = data[(data['Date'] >= '2008-01-01') & (data['Date'] <= '2022-09-01')]

# Prepare the feature matrix X and target vector y for training
X_train = train_data.drop(columns=['Date', target_variable] + high_vif_features)
y_train = train_data[target_variable]

# Hyperparameter grids for tuning
param_grids = {
    'Linear Regression': {},
    'Random Forest': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'bootstrap': [True, False]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Train the models with hyperparameter tuning and save the best hyperparameters
best_params = {}

for name, model in models.items():
    logging.info(f"Tuning and training {name} model.")
    param_grid = param_grids[name]
    if param_grid:  # Skip GridSearchCV if there are no hyperparameters to tune
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[name] = grid_search.best_params_
        logging.info(f"Best params for {name}: {grid_search.best_params_}")
    else:
        model.fit(X_train, y_train)
        best_params[name] = {}

# Save the best hyperparameters to a CSV file
best_params_df = pd.DataFrame(best_params)
best_params_df.to_csv('best_hyperparameters.csv', index=False)

# Print completion message
print("Model training and hyperparameter tuning complete. Hyperparameters saved to CSV.")