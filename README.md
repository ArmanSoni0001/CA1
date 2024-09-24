## Electric Vehicle Charging Station Management

## Overview
This project manages electric vehicle (EV) charging sessions and predicts the energy cost for future charging sessions. It uses historical data to make accurate cost predictions, helping station operators manage their charging infrastructure efficiently.

## Dataset
The dataset includes the following columns:

VehicleID: Unique ID for each vehicle
SessionID: Unique ID for each charging session
Energy_Consumed: Energy consumed during the session
Charging_duration: Duration of the charging session
Cost: Total cost of the charging session

## Machine Learning Models
The project uses three machine learning models to predict charging costs:

Linear Regression
Random Forest
Support Vector Regressor (SVR)
The model with the best performance is selected based on accuracy.

