import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import shap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_vif(data, exclude_columns):
    data_numeric = data.drop(exclude_columns, axis=1)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = data_numeric.columns
    vif_data['VIF'] = [variance_inflation_factor(data_numeric.values, i) for i in range(data_numeric.shape[1])]
    return vif_data

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return mse, r2, mae, rmse



data = pd.read_csv('Monthly_Price_Determinant_Data_v4.csv')
logging.info("Dataset loaded successfully.")

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')

# Define the target variable
target_variable = 'Iron ore, cfr spot ($/dmtu)'

# Shift the target variable down by two rows to align it with the next two months' features
data[target_variable] = data[target_variable].shift(-2)

# Drop the last two rows because they will have NaN after shifting
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

# Prepare the data for prediction (two months into the future)
predict_data = data.tail(1)
print(predict_data)

X_predict = predict_data.drop(columns=['Date', target_variable] + high_vif_features)

# Input the best hyperparameters manually
best_params = {
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'bootstrap': True
    },
    'Gradient Boosting': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_samples_split': 15,
        'min_samples_leaf': 6
    },
    'XGBoost': {
        'n_estimators': 500,
        'learning_rate': 0.05, 
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0.3,
        'subsample': 0.6,
        'colsample_bytree': 0.6  
    }
}

# Initialize the models with best hyperparameters
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(**best_params['Random Forest'], random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(**best_params['Gradient Boosting'], random_state=42),
    'XGBoost': XGBRegressor(**best_params['XGBoost'], random_state=42)
}

# Train the models and calculate cross-validation metrics
best_models = {}
model_metrics = {}
shap_values_dict = {}

for name, model in models.items():
    logging.info(f"Training and evaluating {name} model.")
    best_models[name] = model.fit(X_train, y_train)

    # Cross-validation predictions
    cv_predictions = cross_val_predict(best_models[name], X_train, y_train, cv=5)
    metrics = calculate_metrics(y_train, cv_predictions)
    model_metrics[name] = metrics

    # Skip SHAP calculation for Linear Regression
    if name == 'Linear Regression':
        continue

    # Calculate SHAP values using TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_predict)
    
    # Store the SHAP values in the dictionary
    shap_values_dict[name] = shap_values


# List of models to plot
model_names = ['Random Forest', 'Gradient Boosting', 'XGBoost']

# Generate and save individual SHAP plots
for name in model_names:
    shap_values = shap_values_dict[name]
    
    # Generate a SHAP summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_predict,
        feature_names=X_predict.columns,
        plot_size=[10, 5],
        # plot_type='bar',
        show=False,
        color_bar=False,
    )

    # Save the plot as an image file
    plt.savefig(f'shap_plot_{name}.png')
    plt.close()

# Create a single figure to combine the images
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# Load and display each saved SHAP plot image
for i, name in enumerate(model_names):
    img = mpimg.imread(f'shap_plot_{name}.png')
    axes[i].imshow(img)
    axes[i].axis('off')  # Hide the axes, we only want to show the images
    axes[i].set_title(f"SHAP Summary for {name}", fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the combined figure
plt.show()

# Get predictions
predictions = {}

for name, model in best_models.items():
    logging.info(f"Predicting with {name} model.")
    y_pred = model.predict(X_predict)
    predictions[name] = y_pred

# Adjust the predictions DataFrame to correctly label the date
predictions_df = pd.DataFrame(predictions, index=['2022-12-01'])  # Adjust the date as it's 3 months ahead

# Combine model metrics into a DataFrame and label the columns appropriately
metrics_df = pd.DataFrame(model_metrics, index=['MSE', 'R2', 'MAE', 'RMSE']).T

# Display the model metrics in a clear and organized manner
print("Model Performance Metrics:\n")
print(metrics_df)

# Display weights of coefficients for Linear Regression
if 'Linear Regression' in best_models:
    lr_model = best_models['Linear Regression']
    lr_weights_df = pd.DataFrame(lr_model.coef_, index=X_train.columns, columns=['Coefficient'])
    print("\nLinear Regression Coefficients:\n", lr_weights_df)

# Print the outputs for checking purposes
print("\nPredictions for 2022-12-01:\n", predictions_df)
