# Hourly Traffic Volume Forecasting - Time Series Analysis and Modeling
Individual Project: Rayna Liu

## Background and Introduction
Traffic volume forecasts are used by many transportation analysis and management systems to better characterize and react to fluctuating traffic patterns. Develop a couple of models to forecast the hourly traffic volume and select the best model by observing, analyzing, and comparing these models’ residual and forecast errors.

The purpose of this term project is to find the best model to forecast the hourly traffic volume. First, do the data preprocessing to know the content of the dataset. Then, do the time series analysis to find the dataset's time-series properties, like stationarity, trend, and seasonality. Dealing with data non-stationary, strong seasonality, and strong trends if applicable. Developing multiple models such as Holt-Winter, Multiple Linear Regression, ARMA, ARIMA, SARIMA, and based models. By analyzing their residual and forecast error, the optimal model was selected to forecast traffic volume.

## Dataset
Metro Interstate Traffic Volume Dataset is about hourly Minneapolis-St Paul, MN traffic volume for westbound 1-94. It includes weather and holiday features from 2012-2018 with link [Metro_Interstate_Traffic_Volume.csv](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume).

## Table of Contents of Report
**Hourly Traffic Volume Forecasting Report.pdf**

**I. Description of Dataset**
* Data Preprocessing
* Dependent Variable v.s. Time
* ACF / PACF of Dependent Variable
* Correlation Matrix

**II. Stationary**
* ADF Test
* Plot of Rolling Mean and Variance
* Seasonal / Non-Seasonal Differencing

**III. Time Series Decomposition**
* STL Decomposition Method
* Strength of the Trend and Seasonality
* Plot of Raw Dataset v.s De-trended and Seasonally Adjusted Dataset

**IV. Holt-Winter Method**

**V. Multiple Linear Regression**
* Collinearity Detection
* Feature Reduction
* Hypothesis Tests Analysis
* AIC,BIC,R-squared and Adjusted R-squared
* One-step ahead Forecast
* Residual Analysis

**VI. ARMA, ARIMA, SARIMA**
* Order Determination
* Estimated Parameters of ARMA Model
* Diagnostic Analysis

**VII. Based Models**
* Average Method
* Naive Method
* Seasonal Naive Method
* Drift Method
* Simple Exponential Smoothing
* Holt’s Linear Trend Method

**VIII. Final Model Selection**
* Forecast Function and h-step Prediction

## How It Works
 1. Read the report **Hourly Traffic Volume Forecasting Report.pdf**.
 2. Or save time to watch presentation video, click [Here](https://youtu.be/8lkXXOB94xQ), and read presentation ppt **PPT.pdf**.
 3. Check python code **Code.py** (run **toolbox.py** before running **Code.py**).
