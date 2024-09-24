import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'C:/Users/Arman Soni/OneDrive/Desktop/ev_charging.csv'
charging_data = pd.read_csv(file_path)

# Selecting relevant features and the target variable
features = charging_data[['Energy_consumed', 'Charging_duration']]
target = charging_data['Cost']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()

# Defining models
linear_regression = LinearRegression()
random_forest = RandomForestRegressor(random_state=42)
svr = SVR()

# Creating pipelines for each model
pipeline_lr = Pipeline(steps=[('scaler', scaler), ('linear_regression', linear_regression)])
pipeline_rf = Pipeline(steps=[('scaler', scaler), ('random_forest', random_forest)])
pipeline_svr = Pipeline(steps=[('scaler', scaler), ('svr', svr)])

# Fitting the models
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_svr.fit(X_train, y_train)

# Making predictions
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_svr = pipeline_svr.predict(X_test)

# Evaluating model performance using RMSE and R² score
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
r2_svr = r2_score(y_test, y_pred_svr)

# Displaying the results
print(f"Linear Regression: RMSE = {rmse_lr:.2f}, R² = {r2_lr:.4f}")
print(f"Random Forest: RMSE = {rmse_rf:.2f}, R² = {r2_rf:.4f}")
print(f"Support Vector Regressor: RMSE = {rmse_svr:.2f}, R² = {r2_svr:.4f}")

# Determining the best model
best_model_name = "Random Forest" if (rmse_rf < rmse_lr and rmse_rf < rmse_svr) else (
    "Linear Regression" if (rmse_lr < rmse_svr) else "SVR")

print(f"\nThe best model is: {best_model_name}")
